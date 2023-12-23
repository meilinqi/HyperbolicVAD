import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from utils.loss import InfoNCE


def norm(data):
    l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
    return torch.div(data, l2)


def clas(logits, seq_len):
    logits = logits.squeeze()
    instance_logits = torch.zeros(0).to(logits.device)  # tensor([])
    for i in range(logits.shape[0]):
        if seq_len is None:
            tmp = torch.mean(logits[i]).view(1)
        else:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='floor') + 1),
                                largest=True)
            tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))
    instance_logits = torch.sigmoid(instance_logits)

    return instance_logits

def CENTROPY(logits, logits2, seq_len, device):
    instance_logits = torch.tensor(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        tmp1 = torch.sigmoid(logits[i, :seq_len[i]]).squeeze()
        tmp2 = torch.sigmoid(logits2[i, :seq_len[i]]).squeeze()
        loss = torch.mean(-tmp1.detach() * torch.log(tmp2))
        instance_logits = instance_logits + loss
    instance_logits = instance_logits/logits.shape[0]
    return instance_logits

def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2 - arr) ** 2)

    return lamda1 * loss


class HCIL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hcil_loss = InfoNCE(negative_mode='unpaired')

    def forward(self, abn_score, nor_score, poin_abn_feat, poin_nor_feat, lore_abn_feat, lore_nor_feat, seq_len,
                batch_size, device):
        abn_rep1 = torch.zeros(0).to(device)
        abn_rep2 = torch.zeros(0).to(device)
        nor_rep = torch.zeros(0).to(device)
        abn_seq_len = seq_len[batch_size:]
        nor_seq_len = seq_len[:batch_size]
        abn_logits = torch.sigmoid(abn_score)
        nor_logits = torch.sigmoid(nor_score)

        for i in range(abn_logits.size(0)):
            cur_nor_topk, cur_nor_topk_indices = torch.topk(nor_logits[i][:nor_seq_len[i]], k=int(
                torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
            cur_nor_rep_topk = poin_nor_feat[i][cur_nor_topk_indices]
            cur_dim = cur_nor_rep_topk.size()
            cur_nor_rep_topk = torch.mean(cur_nor_rep_topk, 0, keepdim=True).expand(cur_dim)
            nor_rep = torch.cat((nor_rep, cur_nor_rep_topk), 0)

            # bgd features
            cur_nor_topk, cur_nor_topk_indices = torch.topk(nor_logits[i][:nor_seq_len[i]], k=int(
                torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
            cur_nor_rep_topk = lore_nor_feat[i][cur_nor_topk_indices]
            cur_dim = cur_nor_rep_topk.size()
            cur_nor_rep_topk = torch.mean(cur_nor_rep_topk, 0, keepdim=True).expand(cur_dim)
            nor_rep = torch.cat((nor_rep, cur_nor_rep_topk), 0)

            # cur_nor_inverse_topk, cur_nor_inverse_topk_indices = torch.topk(nor_logits[i][:nor_seq_len[i]], k=int(torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=False)     # return k min score and indices
            # cur_nor_inverse_rep_topk = nor_feat[i][cur_nor_inverse_topk_indices]   #  get min k value
            # bgd_rep = torch.cat((bgd_rep, cur_nor_inverse_rep_topk), 0)

            cur_abn_topk, cur_abn_topk_indices = torch.topk(abn_logits[i][:abn_seq_len[i]], k=int(
                torch.div(abn_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
            cur_abn_rep_topk = poin_abn_feat[i][cur_abn_topk_indices]
            cur_dim = cur_abn_rep_topk.size()
            cur_abn_rep_topk = torch.mean(cur_abn_rep_topk, 0, keepdim=True).expand(cur_dim)
            abn_rep1 = torch.cat((abn_rep1, cur_abn_rep_topk), 0)

            cur_abn_topk, cur_abn_topk_indices = torch.topk(abn_logits[i][:abn_seq_len[i]], k=int(
                torch.div(abn_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
            cur_abn_rep_topk = lore_abn_feat[i][cur_abn_topk_indices]
            cur_dim = cur_abn_rep_topk.size()
            cur_abn_rep_topk = torch.mean(cur_abn_rep_topk, 0, keepdim=True).expand(cur_dim)
            abn_rep2 = torch.cat((abn_rep2, cur_abn_rep_topk), 0)

        min_len, max_len = min(len(abn_rep1), len(nor_rep)), max(len(abn_rep1), len(nor_rep))
        idx = random.sample(range(0, max_len), min_len)
        if len(abn_rep1) > len(nor_rep):
            abn_rep1 = abn_rep1[idx]
            abn_rep2 = abn_rep2[idx]
        else:
            nor_rep = nor_rep[idx]

        if nor_rep.size(0) == 0 or abn_rep1.size(0) == 0 or abn_rep2.size(0) == 0:
            return 0.0
        else:
            loss_a2n = self.hcil_loss(abn_rep1, abn_rep2, nor_rep)
            return loss_a2n


criterion = torch.nn.BCELoss()
HCIL_Loss = HCIL()


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.sigmoid = torch.nn.Sigmoid()
        self.margin = margin

    def forward(self, resout):
        nor_out = resout["nor_out"]
        abn_out = resout["abn_out"]

        loss_abn1 = torch.abs(self.margin - torch.norm(torch.mean(abn_out, dim=1), p=2, dim=1))

        loss_nor1 = torch.norm(torch.mean(nor_out, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn1 + loss_nor1 ) ** 2)

        loss_total = self.alpha * loss_rtfm

        return loss_total

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class HVAD_loss(torch.nn.Module):
    def __init__(self, alpha):
        super(HVAD_loss, self).__init__()
        self.alpha = alpha

        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()
        self.contrastive = ContrastiveLoss()

    def forward(self, resout):
        poin_nor_feat = resout["poin_nor_feat"]
        poin_abn_feat = resout["poin_abn_feat"]
        lore_nor_feat = resout["lore_nor_feat"]
        lore_abn_feat = resout["lore_abn_feat"]

        seperate = len(poin_abn_feat) / 2
        loss_con1 = self.contrastive(torch.norm(poin_abn_feat, p=1, dim=2), torch.norm(poin_nor_feat, p=1, dim=2),
                                    1)  # try tp separate normal and abnormal
        loss_con_n1 = self.contrastive(torch.norm(poin_nor_feat[int(seperate):], p=1, dim=2),
                                      torch.norm(poin_nor_feat[:int(seperate)], p=1, dim=2),
                                      0)  # try to cluster the same class
        loss_con_a1 = self.contrastive(torch.norm(poin_abn_feat[int(seperate):], p=1, dim=2),
                                      torch.norm(poin_abn_feat[:int(seperate)], p=1, dim=2), 0)

        loss_con2 = self.contrastive(torch.norm(lore_abn_feat, p=1, dim=2), torch.norm(lore_nor_feat, p=1, dim=2),
                                    1)  # try tp separate normal and abnormal
        loss_con_n2 = self.contrastive(torch.norm(lore_nor_feat[int(seperate):], p=1, dim=2),
                                      torch.norm(lore_nor_feat[:int(seperate)], p=1, dim=2),
                                      0)  # try to cluster the same class
        loss_con_a2 = self.contrastive(torch.norm(lore_abn_feat[int(seperate):], p=1, dim=2),
                                      torch.norm(lore_abn_feat[:int(seperate)], p=1, dim=2), 0)

        loss_total = self.alpha * (0.001 * loss_con1 + loss_con_a1 + loss_con_n1+0.001 * loss_con2 + loss_con_a2 + loss_con_n2)

        return loss_total

def train(net, normal_loader, abnormal_loader, optimizer, criterion, log_writer, step,args):
    loss = {}
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    inputs = torch.cat((ninput, ainput), 0).float().to(args.device)
    labels = torch.cat((nlabel, alabel), 0).float().to(args.device)
    seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)

    resout = net(inputs, seq_len)
    mil_logits = clas(resout['frame'], seq_len)
    mil_loss = criterion(mil_logits, labels)

    # hcil_loss = HCIL_Loss(predict['abn_score'],
    #                       predict['nor_score'],
    #                       predict['poin_nor_feat'],
    #                       predict['poin_abn_feat'],
    #                       predict['lore_nor_feat'],
    #                       predict['lore_abn_feat'],
    #                       seq_len,
    #                       ninput.size(0),
    #                       inputs.device
    #                       )

    # loss_sparse = sparsity(resout['abn_score'], ainput.size(0), 8e-3)
    # loss_smooth = smooth(torch.sigmoid(resout['frame']), 8e-4)
    # cost = mil_loss + args.lamda * hcil_loss
    #
    # rtfm_criterion = RTFM_loss(0.0001, 100)
    # rtfm_loss = rtfm_criterion(resout)

    # croloss = CENTROPY(logits, logits2, seq_len, device)
    # hvad_loss = HVAD_loss(0.00001)
    # hvad_loss= hvad_loss(resout)
    cost = mil_loss # + rtfm_loss

    loss['total_loss'] = cost
    loss['mil_loss'] = mil_loss
    # loss['hcil_loss'] = hcil_loss
    # loss['loss_sparse'] = loss_sparse
    # loss['loss_smooth'] = loss_smooth
    # loss['triplet_loss'] = resout['triplet_loss']

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    for key in loss.keys():
        log_writer.add_scalar(tag=key, scalar_value=loss[key].item(),global_step=step)