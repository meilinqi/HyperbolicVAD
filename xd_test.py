import pickle

import torch
from matplotlib import pyplot as plt

from options_xd import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def anomap(predict_dict, label_dict, save_root,value):
    path = os.path.join(save_root, 'plot', value)
    if os.path.exists(path) == 0:
        os.makedirs(path)
    for k, v in predict_dict.items():
        predict_np = v.repeat(16)
        label_np = label_dict[k][:len(v.repeat(16))].squeeze()
        x1 = np.arange(len(predict_np))
        x2 = np.arange(len(label_np))
        plt.plot(x1, predict_np, color='b', label='predicted scores', linewidth=1)
        plt.fill_between(x2, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xlabel('Frames')
        plt.ylabel('Anomaly scores')
        plt.grid(True, linestyle='-.')
        plt.legend()
        plt.savefig(os.path.join(path, str(k)+'.png'))
        plt.close()

def test(net, test_loader, step, log_writer=None, test_info=None, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/xd_gt.npy")
        frame_predict = None
        cls_label = []
        cls_pre = []
        predict_dict = {}
        for i in range(len(test_loader.dataset) // 5):

            inputs, labels, name= next(load_iter)
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label.append(int(labels[0]))
            res = net(inputs, None)
            a_predict = torch.sigmoid(res["frame"]).cpu().numpy().mean(0)
            # a_predict = res["frame"].cpu().numpy().mean(0)
            if model_file is not None:
                predict_dict[name[0]] = a_predict
            cls_pre.append(1 if a_predict.max() > 0.5 else 0)
            fpre_ = np.repeat(a_predict, 16)
            if frame_predict is None:
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict, )
        ap_score = auc(recall, precision)
        if log_writer is not None:
            log_writer.add_scalar('roc_auc', auc_score, step)
            log_writer.add_scalar('accuracy', accuracy, step)
            log_writer.add_scalar('pr_auc', ap_score, step)
            # log_writer.add_scalar('scores', frame_predict)
            # log_writer.add_scalar('roc_curve', tpr, fpr)
        if test_info is not None:
            test_info["step"].append(step)
            test_info["auc"].append(auc_score)
            test_info["ap"].append(ap_score)
            test_info["ac"].append(accuracy)
        return {
            'auc_score': auc_score,
            'ap': ap_score,
            'predict_dict': predict_dict
        }


if __name__ == "__main__":
    args = parse_args()
    config = Config(args)
    gpu_idx = args.cuda
    torch.cuda.set_device('cuda:{}'.format(gpu_idx))
    args.device = 'cuda:' + str(gpu_idx) if int(gpu_idx) >= 0 else 'cpu'
    worker_init_fn = None
    config.len_feature = 1024
    if config.seed >= 0:
        util.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    net = Model(args, flag="Test")
    net = net.cuda()
    test_loader = data.DataLoader(
        XDVideo(root_dir=config.root_dir, mode='Test', modal=config.modal, num_segments=config.num_segments,len_feature=config.len_feature),
        batch_size=5,
        shuffle=False, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn)
    file_list = open("./list/XD_Test.list", "r")
    xd_dict = {}
    while True:
        line = file_list.readline()
        if not line:
            break
        if '__0.npy' not in line:
            continue
        tmp = line.split('/')[-1][:-8]
        xd_dict[tmp]=[]
    file_list.close()
    annotations_file = open("./list/XD_Annotation.txt", "r")
    while True:
        line = annotations_file.readline()
        line = line.strip()
        if not line:
            break
        else:
            name_string = line.split(' ')[0]
            frame_string = line.split(' ')[1:]
            frame_int = [ int(t) for t in frame_string]
            xd_dict[name_string] = frame_int
    annotations_file.close()
    res = test(net, test_loader, 0, model_file='models/xd_trans_ap=0.8267633743879315.pkl')
    predict_dict = res['predict_dict']
    for k, v in predict_dict.items():
        s = len(v)*16
        ground_true = np.zeros(s)
        index = xd_dict[k]
        if index:
            for t in range(len(index)//2):
                ground_true[index[t*2]:min(index[t*2+1], s)] = 1
        xd_dict[k] = ground_true
    outputs = os.path.join('outputs', 'xd')
    anomap(predict_dict, xd_dict, outputs, str(res['ap']))
    print(res['ap'])
