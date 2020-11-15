# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D（300个query×每个样本的presentation的维数）
    # y: M x D（60个class×每个分类的presentation的维数）
    n = x.size(0)  # 300
    m = y.size(0)  # 60
    d = x.size(1)  # 576
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')  # 实际的标签
    input_cpu = input.to('cpu')  # ProtoNet计算结果

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)  # 返回有哪些class（returns the unique elements of the input tensor）
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    # 每一个class的n_query+n_support数目都是一样的，所以取第一个class来算就行了，在这个episode里面取到的总样本数，减去n_support就是n_query
    support_idxs = list(map(supp_idxs, classes))
    # 对于每一个class（label），.eq判断target的每一位是否是这个label，.nonzero找出哪几位是target，[:n_support]从里面取出n_support作为支撑集，squeeze将其从n_support*1压缩为n_support的tensor

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])  # 计算每一个label的c_k（representation）
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    # 获取查询集的对应编号，但是最后那个view(-1)啥意思啊

    query_samples = input.to('cpu')[query_idxs]
    # 是按class的顺序排的

    dists = euclidean_dist(query_samples, prototypes)  # (n_query*n_classes)*n_classes

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)  # n_classes*n_query*n_classes

    target_inds = torch.arange(0, n_classes)  # 从0~n_classes-1的tensor
    target_inds = target_inds.view(n_classes, 1, 1)  # n_classes*1*1，内容从0~n_classes-1
    target_inds = target_inds.expand(n_classes, n_query, 1).long()  # n_claasses*n_query*1，从0~n_classes-1每个都有n_query个

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)  # torch.max(Tensor,dim)返回两个值，第一个为储存着最大值的Tensor，第二维为储存着最大值对应的index的Tensor
    # y_hat:n_classes*n_query
    # target_inds:n_classes*n_query*1
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val
