""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import torch.nn as nn
from utils.tools import broadcast_coalesced, all_reduce_coalesced

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay, im_group=None):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        if type(net) == nn.parallel.DistributedDataParallel:
            self.net_in = net
            self.net = self.net_in.module
            self.v_net = copy.deepcopy(net)
        else:
            self.net = net
            self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.im_group = im_group

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
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
        # logger.info("R{} 00 start virtual step".format(rank))
        loss = self.net.loss(trn_X, trn_y)  # L_trn(w)
        # logger.info("R{} 11 end virtual step {}".format(rank, loss))

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())
        # logger.info("R{} 22 end virtual step {}".format(rank, loss))

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            if self.im_group == 'all':
                all_reduce_coalesced(gradients)
            elif self.im_group is not None:
                all_reduce_coalesced(gradients, self.im_group)
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))
            # logger.info("R{} 33 end virtual step {}".format(rank, loss))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)
            # logger.info("R{} 44 end virtual step {}".format(rank, loss))

    def unrolled_backward(self, config_in, trn_X, trn_y, val_X, val_y, xi, w_optim, cor=1):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # init config
        config = config_in
        # logger.info("R{} check 1 {} {}".format(rank, trn_X.shape, trn_y.shape))
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)
        # logger.info("R{} check 2".format(rank))

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)
        # compute gradient
        v_alphas = list(self.v_net.alphas())
        v_arch_weights = list(self.v_net.arch_weights())
        v_nonarch_weights = list(self.v_net.nonarch_weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_arch_weights + v_nonarch_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = list(v_grads[len(v_alphas): len(v_alphas) + len(v_arch_weights)])
        dh = list(v_grads[len(v_alphas) + len(v_arch_weights):])
        if self.im_group == "all":
            all_reduce_coalesced(dw)
            all_reduce_coalesced(dh)
        elif self.im_group is not None:
            all_reduce_coalesced(dw, im_group=self.im_group)
            all_reduce_coalesced(dh, im_group=self.im_group)
        if config.cor == 3:
            # logger.info("R{} check 3.5".format(rank))
            dw2 = broadcast_coalesced(2, dw)
            dw1 = broadcast_coalesced(1, dw)
            # logger.info("R{} check 4".format(rank))
            if config.rank == 0:
                for w, w1, w2 in zip(dw, dw1, dw2):
                    w += w1 * 2 * config.lambda_w
                    w += w2 * pow(2 * config.lambda_w, 2)
            elif config.rank == 1:
                for w, w2 in zip(dw, dw2):
                    w += w2 * 2 * config.lambda_w
            # logger.info("R{} check 4".format(rank))

        hessian = self.compute_hessian(dw, dh, trn_X, trn_y, cor, config)
        # logger.info("R{} check 5".format(rank))
        # clipping hessian
        max_norm = float(config.max_hessian_grad_norm)
        hessian_clip = copy.deepcopy(hessian)
        for h_c, h in zip(hessian_clip, hessian):
            h_norm = torch.norm(h.detach(), dim=-1)
            max_coeff = h_norm / max_norm
            max_coeff[max_coeff < 1.0] = torch.tensor(1.0).cuda(config.gpu)
            h_c = torch.div(h, max_coeff.unsqueeze(-1))
        hessian = hessian_clip
        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            if self.im_group == "all":
                all_reduce_coalesced(dalpha)
                all_reduce_coalesced(hessian)
            elif self.im_group is not None:
                all_reduce_coalesced(dalpha, im_group=self.im_group)
                all_reduce_coalesced(hessian, im_group=self.im_group)
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h
        if cor == 2:
            return dalpha, dw
        return dalpha, hessian

    def compute_hessian(self, dw, dh, trn_X, trn_y, cor, config):
        """
        dw = dw` { L_val(alpha, w`, h`) }, dh = dh` { L_val(alpha, w`, h`) }
        w+ = w + eps_w * dw, h+ = h + eps_h * dh
        w- = w - eps_w * dw, h- = h - eps_h * dh
        hessian_w = (dalpha { L_trn(alpha, w+, h) } - dalpha { L_trn(alpha, w-, h) }) / (2*eps_w)
        hessian_h = (dalpha { L_trn(alpha, w, h+) } - dalpha { L_trn(alpha, w, h-) }) / (2*eps_h)
        eps_w = 0.01 / ||dw||, eps_h = 0.01 / ||dh||
        """
        norm_w = torch.cat([w.view(-1) for w in dw]).norm()
        eps_w = 0.01 / norm_w
        norm_h = torch.cat([h.view(-1) for h in dh]).norm()
        eps_h = 0.01 / norm_h

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.arch_weights(), dw):
                p += eps_w * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_wpos = torch.autograd.grad(loss, self.net.alphas())  # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.arch_weights(), dw):
                p -= 2. * eps_w * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_wneg = torch.autograd.grad(loss, self.net.alphas())  # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.arch_weights(), dw):
                p += eps_w * d

        hessian_w = [(p - n) / (2. * eps_w) for p, n in zip(dalpha_wpos, dalpha_wneg)]

        # h+ = hw + eps*dh`
        with torch.no_grad():
            for p, d in zip(self.net.nonarch_weights(), dh):
                p += eps_h * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_hpos = torch.autograd.grad(loss, self.net.alphas())  # dalpha { L_trn(alpha, w, h+) }

        # h- = h - eps*dh`
        with torch.no_grad():
            for p, d in zip(self.net.nonarch_weights(), dh):
                p -= 2. * eps_h * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_hneg = torch.autograd.grad(loss, self.net.alphas())  # dalpha { L_trn(alpha, w, h-) }

        # recover h
        with torch.no_grad():
            for p, d in zip(self.net.nonarch_weights(), dh):
                p += eps_h * d

        hessian_h = [(p - n) / (2. * eps_h) for p, n in zip(dalpha_hpos, dalpha_hneg)]

        # hessian
        if not cor == 2:
            hessian = [hw + hh for hw, hh in zip(hessian_w, hessian_h)]
        else:
            coef = (1 - pow(2 * config.lambda_w, config.rank+1)) / (1 - 2 * config.lambda_w)
            hessian = [hw * coef + hh for hw, hh in zip(hessian_w, hessian_h)]
        return hessian
