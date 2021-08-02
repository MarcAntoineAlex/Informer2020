from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from models.architect import Architect

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.distributed as dist

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_M_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_M_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
                self.args
            ).float()
        else:
            raise NotImplementedError
        # something

        self.arch = Architect(model, self.device, self.args, self._select_criterion())
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        W_optim = optim.Adam(self.model.W(), lr=self.args.learning_rate)
        A_optim = optim.Adam(self.model.A(), self.args.A_lr, betas=(0.5, 0.999),
                                   weight_decay=self.args.A_weight_decay)
        return W_optim, A_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, val_d in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, val_d)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, ii, logger):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        next_data, next_loader = self._get_data(flag='next')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.path, str(ii))
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, rank=self.args.rank)

        W_optim, A_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # print("*************** {}".format(self.args.rank))
        # print("-----W-------")
        # for n, _ in self.model.named_W():
        #     print(n)
        # print("-----A-------")
        # for n, _ in self.model.named_A():
        #     print(n)
        # print("-----H-------")
        # for n, _ in self.model.named_H():
        #     print(n)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (trn_data, val_data, next_data) in enumerate(zip(train_loader, vali_loader, next_loader)):
                for i in range(len(trn_data)):
                    trn_data[i], val_data[i], next_data[i] = trn_data[i].float().to(self.device), val_data[i].float().to(self.device), next_data[i].float().to(self.device)
                iter_count += 1
                # A_optim.zero_grad()
                # self.arch.unrolled_backward(self.args, trn_data, val_data, next_data, W_optim.param_groups[0]['lr'], W_optim)
                # for r in range(1, self.args.world_size):
                #     for n, h in self.model.named_H():
                #         if "proj.{}".format(r) in n:
                #             if self.args.rank <= r:
                #                 with torch.no_grad():
                #                     dist.all_reduce(h.grad)
                #                     h.grad *= self.args.world_size/r+1
                #             else:
                #                 z = torch.zeros(h.shape).to(self.device)
                #                 dist.all_reduce(z)
                # A_optim.step()

                W_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, trn_data)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info("\tR{0} iters: {1}, epoch: {2} | loss: {3:.7f}".format(self.args.rank, i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(W_optim)
                    scaler.update()
                else:
                    loss.backward()
                    W_optim.step()

            logger.info("R{} Epoch: {} cost time: {}".format(self.args.rank, epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            logger.info("R{0} Epoch: {1}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                self.args.rank, epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(W_optim, epoch + 1, self.args)

        best_model_path = path + '/' + '{}_checkpoint.pth'.format(self.args.rank)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, logger):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, test_d in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, test_d)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        logger.info('test shape: {} {}'.format(preds.shape, trues.shape))
        preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
        trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))
        logger.info('test shape: {} {}'.format(preds.shape, trues.shape))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logger.info('R{} mse:{}, mae:{}'.format(self.args.rank, mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, pred_d in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, pred_d)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, data):
        batch_x = data[0].float().to(self.device)
        batch_y = data[1].float().to(self.device)

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
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
