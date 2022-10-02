import torch.nn as nn
import resnet
import time
import numpy as np
import sys
#import scipy.io as scio
import torch
import torchvision.transforms as transforms
import torchvision
from torch import nn
from torch.nn import functional
from multiprocessing import cpu_count

# TODO Design the detector.
# tips: Use pretrained `resnet` as backbone.
class Detector(nn.Module):

    def __init__(self,backbone='resnet50', lengths=(2048 * 4 * 4, 2048, 512),
                     **kwargs):
        super(Detector,self).__init__()
        self.Res = resnet.resnet18(last_fc = True,**kwargs)
        # 虽然这里标的是resnet18但是我把resnet class实现改了一下，现在它只有10层，平均验证精度提高了13%左右

    # , block = resnet.Bottleneck
    def __call__(self,X):
        return self.Res.forward(X)

    def train(self):
        self.Res.train()
        return
    def eval(self):
        self.Res.eval()
        return



...

# def train(net, tr_iter, te_iter, batch_size, optimizer,
#           loss=nn.CrossEntropyLoss(),
#           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#           num_epochs=100):
#     """
#     The train function.
#     """
#     net = net.to(device)
#     temp_batch_count = 0
#     print("Training on", device)
#     for epoch in range(num_epochs):
#         temp_tr_loss_sum, temp_tr_acc_sum, temp_num, temp_start_time = 0., 0., 0, time.time()
#         for x, y in tr_iter:
#             x = x.to(device)
#             y = y.to(device)
#             temp_y_pred = net(x)
#             temp_loss = loss(temp_y_pred, y)
#             optimizer.zero_grad()
#             temp_loss.backward()
#             optimizer.step()
#             temp_tr_loss_sum += temp_loss.cpu().item()
#             temp_tr_acc_sum += (temp_y_pred.argmax(dim=1) == y).sum().cpu().item()
#             temp_num += y.shape[0]
#             temp_batch_count += 1
#         test_acc = evaluate_accuracy(te_iter, net)
#         print("Epoch %d, loss %.4f, training acc %.3f, test ass %.3f, time %.1f s" %
#               (epoch + 1, temp_tr_loss_sum / temp_batch_count, temp_tr_acc_sum / temp_num, test_acc,
#                time.time() - temp_start_time))
#
#
# def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     """
#     The evaluate function, and the performance measure is accuracy.
#     """
#     ret_acc, temp_num = 0., 0
#     with torch.no_grad():
#         for x, y in data_iter:
#             net.eval() # The evaluate mode, and the dropout is closed.
#             ret_acc += (net(x.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
#             net.train()
#             temp_num += y.shape[0]
#
#     return ret_acc / temp_num
# End of todo
