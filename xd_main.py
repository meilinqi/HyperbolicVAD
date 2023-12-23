import pdb
import numpy as np
import torch.utils.data as data
from tensorboardX import SummaryWriter

import util
from optim import RiemannianAdam, RiemannianSGD
from options_xd import *
from config import *
from train import *
from xd_test import test
from model import *
from util import Visualizer
import time
from dataset_loader import *
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None
    timestr = time.strftime("%Y%m%d-%H%M%S")

    gpu_idx = args.cuda
    torch.cuda.set_device('cuda:{}'.format(gpu_idx))
    # args.device = 'cuda:' + str(gpu_idx) if int(gpu_idx) >= 0 else 'cpu'
    args.device = 'cuda'
    if config.seed >= 0:
        util.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    
    config.len_feature = 1024
    net = Model(args, flag="Train")
    if torch.cuda.is_available():
        net = net.to(args.device)
    # net = net.cuda()

    normal_train_loader = data.DataLoader(
        XDVideo(root_dir = config.root_dir, mode = 'Train',modal = config.modal, num_segments = args.num_segments, len_feature = config.len_feature, is_normal = True),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = data.DataLoader(
        XDVideo(root_dir = config.root_dir, mode='Train', modal = config.modal, num_segments = args.num_segments, len_feature = config.len_feature, is_normal = False),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        XDVideo(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 5,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[]}
    
    best_ap = 0

    criterion = torch.nn.BCELoss()
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
    # optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
    #     betas = (0.9, 0.999), weight_decay = 0.00005)
    # wind = Visualizer(env='XD_URDMU', server='http://127.0.0.1', port=8097)

    print(">>> training params: {:.3f}M".format(
        sum(p.numel() for p in net.parameters() if p.requires_grad) / 1000000.0))
    # print(">>> FLOPs: {:.3f}G".format(flops.total() / 1000000000.0))
    print("==========================================\n")
    output_path = 'logs'   # put your own path here
    log_writer = SummaryWriter(output_path)
    # test(net, test_loader,0, log_writer=log_writer, test_info=test_info)
    txt_path = os.path.join(config.output_path, 'xd', 'txt')
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
        train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion, log_writer, step, args)
        if step % 10 == 0 and step > 200:
            test(net, test_loader, step, log_writer=log_writer, test_info=test_info)
            lr_scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {cur_lr}")
            print("ap:" + str(test_info["ap"][-1]))
            if test_info["ap"][-1] > best_ap:
                best_ap = test_info["ap"][-1]
                print("best_ap:" + str(best_ap))
                util.save_best_record(test_info,
                    os.path.join(txt_path, "xd_best_record_ap={}.txt".format(str(best_ap))), args)

                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "xd_trans_ap={}.pkl".format(best_ap)))
            if step == config.num_iters:
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "xd_trans_{}.pkl".format(step)))
    info_path=os.path.join(config.output_path, 'xd_info')
    if os.path.exists(info_path) == 0:
        os.makedirs(info_path)
    util.save_info(test_info, os.path.join(info_path, 'c={:.4f}.txt'.format(args.c)))
    print("best_ap:" + str(best_ap))

