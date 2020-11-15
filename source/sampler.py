# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        # print(labels,classes_per_it,num_samples,iterations)
        self.classes, self.counts = np.unique(self.labels, return_counts=True)  # 每个标签有多少个数量
        # 类，与类的数量
        self.classes = torch.LongTensor(self.classes)
        # 从np转换成tensor！

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))  # 样本数量的iterator
        # print("idxs",self.idxs)
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)  # 与classes尺寸一样的全零tensor
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            # print(label,label_idx)
            # int到int的转换，可以无视（...）
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            # 把第一个是nan的替换成idx，最终label_idx对应的行就是属于这个label的所有数据
            self.numel_per_class[label_idx] += 1
            # label_idx总共有多少个属于它的数据

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class  # support的数量+query的数量（5+5）
        cpi = self.classes_per_it  # N_c（60）

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]  # 取出cpi（N_c）个label
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # print("s",s)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()  # 抽取哪一行（哪个label）  译者注owo:找到classes中标号为c的元素下标
                # print("label_idx",label_idx)
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]  # 取出spc个样本
                # randperm(n): 返回一个随机排序过的0~n-1的数组
                # 从这列里随机拿出需要的个数的数据
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
