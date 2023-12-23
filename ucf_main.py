import pdb
import numpy as np
import torch.utils.data as data
from tensorboardX import SummaryWriter

import util
from optim import RiemannianAdam, RiemannianSGD
from options_ucf import *
from config import *
from train import *
from ucf_test import test
from model import *
from util import Visualizer
import os
import time
from dataset_loader import *
from tqdm import tqdm

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    config = Config(args)
    worker_init_fn = None
    timestr = time.strftime("%Y%m%d-%H%M%S")
    gpu_idx = args.cuda
    torch.cuda.set_device('cuda:{}'.format(gpu_idx))
    args.device = 'cuda:' + str(gpu_idx) if int(gpu_idx) >= 0 else 'cpu'
    if config.seed >= 0:
        util.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    config.len_feature = 1024
    net = Model(args, flag="Train")
    net = net.cuda()
    normal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = args.num_segments, len_feature = config.len_feature, is_normal = True),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = args.num_segments, len_feature = config.len_feature, is_normal = False),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)
    test_info = {"step": [], "auc": [], "ap":[], "ac":[]}
    best_auc = 0
    criterion = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
    #     betas = (0.9, 0.999), weight_decay = 0.00005)

    no_decay = ['bias', 'scale']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in net.named_parameters()
            if p.requires_grad and not any(
                nd in n
                for nd in no_decay) and not isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
            args.weight_decay
    }, {
        'params': [
            p for n, p in net.named_parameters() if p.requires_grad and any(
                nd in n
                for nd in no_decay) or isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
            0.0
    }]
    if args.optimizer == 'radam':
        optimizer = RiemannianAdam(params=optimizer_grouped_parameters,
                                   lr=args.lr,
                                   stabilize=10)
    elif args.optimizer == 'rsgd':
        optimizer = RiemannianSGD(params=optimizer_grouped_parameters,
                                  lr=args.lr,
                                  stabilize=10)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=int(
                                                       args.lr_reduce_freq),
                                                   gamma=float(args.gamma))
    output_path = 'logs'
    log_writer = SummaryWriter(output_path)
    # wind = Visualizer(env='ucf_main_v1', server='http://127.0.0.1', port=8097)
    test(net, test_loader,0,log_writer,test_info)
    txt_path = os.path.join(config.output_path, 'ucf', 'txt')
    if os.path.exists(txt_path) == 0:
        os.makedirs(txt_path)
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        # if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
        train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion, log_writer, step,args)

        if step % 10 == 0 and step > 10:
            test(net, test_loader, step, log_writer, test_info )
            lr_scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {cur_lr}")
            print("auc:" + str(test_info["auc"][-1]))

            if test_info["auc"][-1] > best_auc:
                best_auc = test_info["auc"][-1]
                print("best_auc:" + str(best_auc))

                util.save_best_record(test_info,
                                      os.path.join(txt_path, "ucf_best_record_auc={}.txt".format(str(best_auc))),
                                      args)

                torch.save(net.state_dict(), os.path.join(args.model_path, \
                                                          "ucf_trans_auc={}.pkl".format(best_auc)))
            if step == config.num_iters:
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "ucf_trans_{}.pkl".format(step)))
    print("best_auc:" + str(best_auc))

