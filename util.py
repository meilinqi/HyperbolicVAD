import math
import torch
import numpy as np
import random
import visdom

class Visualizer(object):
    def __init__(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y = np.array([y]), X = np.array([x]),
                      win = str(name),
                      opts = dict(title=name),
                      update = None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img = img, win = name, opts = dict(title = name))
    def lines(self, name, line, X = None):
        if X is None:
            self.vis.line(Y = line, win = name)
        else:
            self.vis.line(X = X, Y = line, win = name)
    def scatter(self, name, data):
        self.vis.scatter(X = data, win = name)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_perturb(feature_len, length):
    r = np.linspace(0, feature_len, length + 1, dtype = np.uint16)
    return r

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)
    
def save_best_record(test_info, file_path,args):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("auc: {:.4f}\n".format(test_info["auc"][-1]))
    fo.write("ap: {:.4f}\n".format(test_info["ap"][-1]))
    fo.write("ac: {:.4f}\n".format(test_info["ac"][-1]))

    fo.write("lamda: {:.4f}\n".format(args.lamda))
    if args.c is not None:
        fo.write("c: {:.4f}\n".format(args.c))
    else:
        fo.write("c: None\n")
    fo.write("gamma: {:.4f}\n".format(args.gamma))
    fo.write("lr_reduce_freq: {}\n".format(args.lr_reduce_freq))
    fo.write("lr: {:.4f}\n".format(args.lr))
    fo.write("num_segments: {}\n".format(args.num_segments))
    fo.write("seed: {}\n".format(args.seed))
    fo.write("use_att: {}\n".format(args.use_att))
    fo.write("num_layers: {}\n".format(args.num_layers))

def save_info(test_info,path):
    with open(path, 'w') as f:
        for v in test_info["step"]:
            f.write("{} ".format(v))
        f.write('\n')
        for v in test_info["auc"]:
            f.write("{:.4f} ".format(v))
        f.write('\n')
        for v in test_info["ap"]:
            f.write("{:.4f} ".format(v))
        f.close()




