import shutil
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


def count_node(model):
    '''
    输入模型，统计非零数量
    '''
    alive_sum, node_sum = 0, 0  # 存活数量和总数量
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            alive_sum += torch.sum(m.weight != 0).item()
            node_sum += m.weight.data.numel()
    print('当前稀疏度', round(alive_sum / node_sum * 100, 5))


def prune_by_mask(model):
    # 根据自己的掩码执行更新
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.mul_(m.weight.mask)


def update_mask(model, percent):
    # 更新Conv2d的掩码
    update_mask = 0  # 总数量
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            update_mask += m.weight.data.numel()
    conv_weights = torch.zeros(update_mask, requires_grad=False)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)
                         ] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(len(conv_weights) * percent)
    thre = y[thre_index]

    pruned = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            m.weight.mask.copy_(mask)
            pruned = pruned + mask.numel() - torch.sum(mask)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
