""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import torch.nn as nn
from utils.tools import broadcast_coalesced, all_reduce_coalesced
import torch.distributed as dist
import utils.tools as tools

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, device, args, criterion, inverse_transform=None):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.device = device
        self.args = copy.deepcopy(args)
        self.criterion = criterion
        if type(net) == nn.parallel.DistributedDataParallel:
            self.net_in = net
            self.net = self.net_in.module
            self.v_net = copy.deepcopy(net)
        else:
            self.net = net.to(self.device)
            self.v_net = copy.deepcopy(net)
        self.w_momentum =self.args.w_momentum
        self.w_weight_decay = self.args.w_weight_decay
        if self.args.inverse:
            self.inverse_transform = inverse_transform

    def virtual_step(self, trn_data, next_data, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        pred = torch.zeros(trn_data[1][:, -self.args.pred_len:, :].shape).to(self.device)
        if self.args.rank == 0:
            pred, true = self._process_one_batch(trn_data, self.net)
            loss = self.criterion(pred, true)
            gradients = torch.autograd.grad(loss, self.net.W())
            with torch.no_grad():
                for w, vw, g in zip(self.net.W(), self.v_net.W(), gradients):
                    m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                    vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                for a, va in zip(self.net.A(), self.v_net.A()):
                    va.copy_(a)
        for r in range(0, self.args.world_size-1):
            if self.args.rank == r:
                pred, true = self._process_one_batch(next_data, self.v_net)
            dist.broadcast(pred.contiguous(), r)
            if self.args.rank == r+1:
                trn_data[1] = torch.cat([trn_data[1][:, :self.args.label_len, :], pred], dim=1)
                pred, true = self._process_one_batch(trn_data, self.net)
                loss = self.criterion(pred, true)
                gradients = torch.autograd.grad(loss, self.net.W())
                with torch.no_grad():
                    for w, vw, g in zip(self.net.W(), self.v_net.W(), gradients):
                        m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                        vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                    for a, va in zip(self.net.A(), self.v_net.A()):
                        va.copy_(a)
        return trn_data[1]

    def unrolled_backward(self, args_in, trn_data, val_data, next_data, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # init config
        args = args_in
        # do virtual step (calc w`)
        trn_data[1] = self.virtual_step(trn_data, next_data, xi, w_optim)

        # calc unrolled loss
        pred, true = self._process_one_batch(val_data, self.v_net)
        loss = self.criterion(pred, true)

        # compute gradient
        v_H = list(self.v_net.H())
        v_W = list(self.v_net.W())
        v_grads = list(torch.autograd.grad(loss, v_W + v_H))
        dw = v_grads[:len(v_W)]
        dH = list(v_grads[len(v_W):])

        hessian = self.compute_hessian(dw, trn_data, next_data, args)
        # logger.info("R{} check 5".format(rank))
        # clipping hessian
        max_norm = float(args.max_hessian_grad_norm)
        hessian_clip = copy.deepcopy(hessian)
        for n, (h_c, h) in enumerate(zip(hessian_clip, hessian)):
            h_norm = torch.norm(h.detach(), dim=-1)
            max_coeff = h_norm / max_norm
            max_coeff[max_coeff < 1.0] = torch.tensor(1.0).cuda(args.gpu)
            hessian_clip[n] = torch.div(h, max_coeff.unsqueeze(-1))
        hessian = hessian_clip
        zero_list = []
        for i, (n, p) in enumerate(self.net.named_H()):
            for k in range(0, max(self.args.rank, 1)):
                if "proj.{}".format(k) in n:
                    zero_list.append(i)
        for i in zero_list:
            dH[i] *= 0
        # update final gradient = dalpha - xi*hessian
        rate_counter = tools.AverageMeter()
        with torch.no_grad():
            for (n, h), dh, he in zip(self.net.named_H(), dH, hessian):
                h.grad = 0
                if he.norm().item() != 0 :
                    rate_counter.update(dh.norm().item()/he.norm().item()/xi)
        return rate_counter.avg


    def compute_hessian(self, dw, trn_data, next_data, args):
        """
        dw = dw` { L_val(alpha, w`, h`) }, dh = dh` { L_val(alpha, w`, h`) }
        w+ = w + eps_w * dw, h+ = h + eps_h * dh
        w- = w - eps_w * dw, h- = h - eps_h * dh
        hessian_w = (dalpha { L_trn(alpha, w+, h) } - dalpha { L_trn(alpha, w-, h) }) / (2*eps_w)
        hessian_h = (dalpha { L_trn(alpha, w, h+) } - dalpha { L_trn(alpha, w, h-) }) / (2*eps_h)
        eps_w = 0.01 / ||dw||, eps_h = 0.01  ||dh||
        """
        norm_w = torch.cat([w.view(-1) for w in dw]).norm()
        eps_w = 0.01 / norm_w
        trn_data[1].requires_grad = True

        zero_list, zero_list2 = [], []
        for i, (n, p) in enumerate(self.net.named_H()):
            for j in range(0, self.args.rank+1):
                if "proj.{}".format(j) in n:
                    zero_list2.append(i)
            for k in range(0, max(self.args.rank, 1)):
                if "proj.{}".format(k) in n:
                    zero_list.append(i)

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p += eps_w * d
        pred, true = self._process_one_batch(trn_data, self.net)
        loss = self.criterion(pred, true)
        HD = list(self.net.H())
        HD.append(trn_data[1])
        d_wpos = list(torch.autograd.grad(loss, HD))
        dH_wpos = d_wpos[:-1]
        dD_wpos = d_wpos[-1][:, -self.args.pred_len:, :].contiguous()
        dD_wposs = [torch.zeros(dD_wpos.shape).to(self.device) for i in range(args.world_size)]
        dist.all_gather(dD_wposs, dD_wpos)
        if args.rank < args.world_size-1:
            pred, _ = self._process_one_batch(next_data, self.v_net)
            pseudo_loss = (pred*dD_wposs[args.rank+1]).sum()
            dH2_wpos = list(torch.autograd.grad(pseudo_loss, self.v_net.H()))
            for i in zero_list2:
                dH2_wpos[i] *= 0
        for i in zero_list:
            dH_wpos[i] *= 0


        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p -= 2. * eps_w * d
        pred, true = self._process_one_batch(trn_data, self.net)
        loss = self.criterion(pred, true)
        d_wneg = list(torch.autograd.grad(loss, HD))
        dH_wneg = d_wneg[:-1]
        dD_wneg = d_wneg[-1][:, -self.args.pred_len:, :].contiguous()
        dD_wnegs = [torch.zeros(dD_wneg.shape).to(self.device) for i in range(args.world_size)]
        dist.all_gather(dD_wnegs, dD_wneg)
        if args.rank < args.world_size-1:
            pred, _ = self._process_one_batch(next_data, self.v_net)
            pseudo_loss = (pred*dD_wnegs[args.rank+1]).sum()
            dH2_wneg = list(torch.autograd.grad(pseudo_loss, self.v_net.H()))
            for i in zero_list2:
                dH2_wneg[i] *= 0
        for i in zero_list:
            dH_wneg[i] *= 0

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p += eps_w * d

        if args.rank < args.world_size-1:
            hessian = [(p - n + p2 - n2) / (2. * eps_w) for p, n, p2, n2 in zip(dH_wpos, dH_wneg, dH2_wpos, dH2_wneg)]
        else:
            hessian = [(p - n) / (2. * eps_w) for p, n in zip(dH_wpos, dH_wneg)]
        return hessian

    def _process_one_batch(self, data, model):
        batch_x = data[0].float().to(self.device)
        batch_y = data[1].float()

        batch_x_mark = data[2].float().to(self.device)
        batch_y_mark = data[3].float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = self.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
