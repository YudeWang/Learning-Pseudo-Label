# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import random
import torchvision
import cv2
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import config_dict
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
from net.sync_batchnorm.replicate import patch_replication_callback
from torch.utils.data import DataLoader
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.DenseCRF import dense_crf
from utils.test_utils import multi_gpu_test, single_gpu_test
from utils.imutils import onehot

torch.manual_seed(1) # cpu
torch.cuda.manual_seed(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def ClassLogSoftMax(f, category):
	exp = torch.exp(f)
	exp_norm = exp/torch.sum(exp*category, dim=1, keepdim=True)
	softmax = exp_norm*category
	logsoftmax = torch.log(exp_norm)*category
	return softmax, logsoftmax

def ReducePivot(x, pivot_num):
	n,c,h,w = x.size()
	x = x.view(n,c//pivot_num,pivot_num,h,w)
	x = torch.max(x,dim=2,keepdim=False)[0]
	return x

def inference(cfg, pseudopath=None):
	dataset = generate_dataset(cfg, period='train', transform='none')
	def worker_init_fn(worker_id):
		np.random.seed(1 + worker_id)
	dataloader = DataLoader(dataset, 
				batch_size=1, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS,
				worker_init_fn = worker_init_fn)
	
	net = generate_net(cfg, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
	print('net initialize')

	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT)
	net.load_state_dict(model_dict, strict=False)

	print('Use %d GPU'%cfg.GPUS)
	assert torch.cuda.device_count() == cfg.GPUS
	device = torch.device('cuda')
	net.to(device)
	net.eval()

	def prepare_func(sample):	
		image_msf = []
		for rate in cfg.TEST_MULTISCALE:
			inputs_batched = sample['image_%f'%rate]
			image_msf.append(inputs_batched)
			if cfg.TEST_FLIP:
				image_msf.append(torch.flip(inputs_batched,[3]))
		return image_msf

	def inference_func(model, img):
		seg = model(img)
		return seg

	def collect_func(result_list, sample):
		[batch, channel, height, width] = sample['image'].size()
		for i in range(len(result_list)):
			result_seg = F.interpolate(result_list[i], (height, width), mode='bilinear', align_corners=True)	
			if cfg.TEST_FLIP and i % 2 == 1:
				result_seg = torch.flip(result_seg, [3])
			result_list[i] = result_seg

		pred = torch.cat(result_list, dim=0)
		pred = torch.mean(pred, dim=0, keepdim=True)
		gt = sample['category'].float().to(0)
		prob_seg = ClassLogSoftMax(pred, gt)[0][0]

		pseudo_label = torch.argmax(prob_seg, dim=0, keepdim=False).cpu().numpy()
		return pred[0].cpu().numpy(), pseudo_label

	def save_step_func(result_sample):
		pseudo_label = {'name': result_sample['name'], 'predict':result_sample['predict'][1]}
		dataset.save_pseudo_gt([pseudo_label], pseudopath)

	result_list = single_gpu_test(net, dataloader, prepare_func=prepare_func, inference_func=inference_func, collect_func=collect_func, save_step_func=save_step_func)
	print('Inference finished')

if __name__ == '__main__':
	cfg = Configuration(config_dict, False)
	inference(cfg)


