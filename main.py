
# basic
import copy
import os
import socket
import time
import warnings
import copy
import random

# torch
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.utils.data as data

# informer
import utils.tools as tools
from exp.exp_m import Exp_M_Informer


def main():
    config = tools.setup()
    ngpus_per_node = torch.cuda.device_count()
    config.ngpus_per_node = ngpus_per_node
    if config.mp_dist:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.nnode = config.world_size
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # worker process function
        mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call worker function on first GPU device
        worker(None, ngpus_per_node, config)


def worker(gpu, ngpus_per_node, args_in):
    # init
    args = copy.deepcopy(args_in)
    jobid = os.environ["SLURM_JOBID"]
    procid = int(os.environ["SLURM_PROCID"])
    args.gpu = gpu

    if args.gpu is not None:
        logger_name = "{}.{}-{:d}-{:d}.search.log".format(args.name, jobid, procid, gpu)
    else:
        logger_name = "{}.{}-{:d}-all.search.log".format(args.name, jobid, procid)

    logger = tools.get_logger(os.path.join(args.path, logger_name))

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])

    if args.mp_dist:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu

    args.print_params(logger.info)

    # get cuda device
    device = torch.device('cuda', gpu)

    # begin
    logger.info("Logger is set - training start")


    logger.info(
        'back:{}, dist_url:{}, world_size:{}, rank:{}'.format(args.dist_backend, args.dist_url, args.world_size,
                                                              args.rank))
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    Exp = Exp_M_Informer

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                    args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.d_model,
                    args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed,
                    args.distil, args.mix, args.des, ii)

        exp = Exp(args)  # set experiments
        logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(ii, logger)

        logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, logger)

        if args.do_predict:
            logger.info('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()