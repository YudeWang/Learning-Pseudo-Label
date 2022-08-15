#=========================================
# Written by Yude Wang
#=========================================
import torch
import torch.nn as nn
import torch.nn.functional as F

def ClassLogSoftMax(f, category, T=1):
	exp = torch.exp(f/T)
	exp_sum = torch.sum(exp*category, dim=1, keepdim=True)+1e-5
	exp_norm = exp/exp_sum
	softmax = exp_norm*category
	logsoftmax = torch.log(exp_norm+1e-5)#*category
	return softmax, logsoftmax
	
def SegLoss(clslogsoftmax, label, weight=None, ignore_idx=255, reduction='mean'):
	loss = F.nll_loss(clslogsoftmax, label, weight=weight, ignore_index=ignore_idx, reduction=reduction)
	return loss

