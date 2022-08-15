# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
import time
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *
from utils.imutils import *
from utils.registry import DATASETS

class BaseDataset(Dataset):
	def __init__(self, cfg, period, transform='none'):
		super(BaseDataset, self).__init__()
		self.cfg = cfg
		self.period = period
		self.transform = transform
		if 'train' not in self.period:
			assert self.transform == 'none'
		self.num_categories = None
		self.totensor = ToTensor()
		self.imagenorm = ImageNorm(cfg.DATA_MEAN, cfg.DATA_STD)
		
		if self.transform != 'none':
			if cfg.DATA_RANDOMCROP > 0:
				self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
			if cfg.DATA_RANDOMSCALE != 1:
				self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
			if cfg.DATA_RANDOMFLIP > 0:
				self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
			if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
				self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)

			#if self.transform == 'strong':
			self.img_aug_ops = []
			if self.cfg.DATA_RANDOMAUTOCONTRAST is not None:
				self.img_aug_ops.append(RandomAutoContrast(self.cfg.DATA_RANDOMAUTOCONTRAST))
			if self.cfg.DATA_RANDOMBRIGHTNESS is not None:
				self.img_aug_ops.append(RandomBrightness(self.cfg.DATA_RANDOMBRIGHTNESS))
			if self.cfg.DATA_RANDOMCOLOR is not None:
				self.img_aug_ops.append(RandomColor(self.cfg.DATA_RANDOMCOLOR))
			if self.cfg.DATA_RANDOMCONTRAST is not None:
				self.img_aug_ops.append(RandomContrast(self.cfg.DATA_RANDOMCONTRAST))
			if self.cfg.DATA_RANDOMEQUALIZE is not None:
				self.img_aug_ops.append(RandomEqualize(self.cfg.DATA_RANDOMEQUALIZE))
			if self.cfg.DATA_RANDOMINVERT is not None:
				self.img_aug_ops.append(RandomInvert(self.cfg.DATA_RANDOMINVERT))
			if self.cfg.DATA_RANDOMPOSTERIZE is not None:
				self.img_aug_ops.append(RandomPosterize(self.cfg.DATA_RANDOMPOSTERIZE))
			if self.cfg.DATA_RANDOMSHARPNESS is not None:
				self.img_aug_ops.append(RandomSharpness(self.cfg.DATA_RANDOMSHARPNESS))
			if self.cfg.DATA_RANDOMSOLARIZE is not None:
				self.img_aug_ops.append(RandomSolarize(self.cfg.DATA_RANDOMSOLARIZE))

			self.both_aug_ops = [self.randomscale, self.randomflip]
			if self.cfg.DATA_RANDOMSHEARX is not None and self.cfg.DATA_RANDOMSHEARY is not None:
				self.both_aug_ops.append(RandomShear(self.cfg.DATA_RANDOMSHEARX, self.cfg.DATA_RANDOMSHEARY, self.cfg.DATA_MEAN))
			if self.cfg.DATA_RANDOMROTATION is not None:
				self.both_aug_ops.append(RandomRotation(cfg.DATA_RANDOMROTATION, self.cfg.DATA_MEAN))
			if self.cfg.DATA_RANDOMCUTOUT is not None:
				self.both_aug_ops.append(RandomCutout(self.cfg.DATA_RANDOMCUTOUT, self.cfg.DATA_MEAN))

			if cfg.DATA_RANDOMCOPYPASTE>0:
				self.randomcopypaste = RandomCopyPaste(num=1)
		else:
			self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)


	def __getitem__(self, idx):
		sample = self.__sample_generate__(idx)

		if self.cfg.DATA_RANDOMMIXUP and self.transform != 'none':
			idx2 = random.randint(0, len(self.name_list)-1)
			sample2 = self.__sample_generate__(idx2)
			sample = self.__mix__(sample, sample2)

		if self.cfg.DATA_RANDOMCOPYPASTE>0 and self.transform != 'none':
			for i in range(self.cfg.DATA_RANDOMCOPYPASTE):
				idx2 = random.randint(0, len(self.name_list)-1)
				sample2 = self.__sample_generate__(idx2)
				#sample = self.randomcopypaste(sample, sample2, t='labeled')
				sample = self.randomcopypaste(sample, sample2, t=self.cfg.DATA_RANDOMCOPYPASTE_TYPE)

		if 'segmentation' in sample.keys():
			sample['mask'] = sample['segmentation'] < self.num_categories
			t = sample['segmentation'].copy()
			t[t >= self.num_categories] = 0
			sample['segmentation_onehot']=onehot(t,self.num_categories)
		return self.totensor(sample)

	def __sample_generate__(self, idx, split_idx=0):
		name = self.load_name(idx)
		image = self.load_image(idx)
		r,c,_ = image.shape
		sample = {'image': image, 'name': name, 'row': r, 'col': c, 'image_orig': image}

		if 'test' in self.period:
			return self.__transform__(sample)
		elif self.cfg.DATA_PSEUDO_GT and idx>=split_idx and 'train' in self.period:
			segmentation = self.load_pseudo_segmentation(idx)
		else:
			segmentation = self.load_segmentation(idx)
		sample['segmentation'] = segmentation
		t = sample['segmentation'].copy()
		t[t >= self.num_categories] = 0
		sample['category'] = seg2cls(t,self.num_categories)
		sample['category_copypaste'] = np.zeros(sample['category'].shape)

		if self.transform != 'strong' and self.cfg.DATA_FEATURE_DIR:
			feature = self.load_feature(idx)
			sample['feature'] = feature
		sample = self.__transform__(sample)
		return sample
		

	def __transform__(self, sample):
		if self.transform == 'weak':
			sample = self.__weak_augment__(sample)
		elif self.transform == 'strong':
			sample = self.__strong_augment__(sample)
		else:
			sample = self.imagenorm(sample)
			sample = self.multiscale(sample)
		return sample

	def __weak_augment__(self, sample):
		if self.cfg.DATA_RANDOM_H>0 or self.cfg.DATA_RANDOM_S>0 or self.cfg.DATA_RANDOM_V>0:
			sample = self.randomhsv(sample)
		if self.cfg.DATA_RANDOMFLIP > 0:
			sample = self.randomflip(sample)
		if self.cfg.DATA_RANDOMSCALE != 1:
			sample = self.randomscale(sample)
		sample['image_strong'] = self.__image_strong_augment__(sample['image'].copy())
		sample = self.imagenorm(sample)
		if self.cfg.DATA_RANDOMCROP > 0:
			sample = self.randomcrop(sample)
		return sample

	def __strong_augment__(self, sample):
		sample['image'] = Image.fromarray(sample['image'])
		time_img = random.randint(0,self.cfg.DATA_AUGTIME)
		ops = random.choices(self.img_aug_ops, k=time_img)
		for op in ops:
			sample = op(sample)
		sample['image'] = np.asarray(sample['image'])
		ops = random.choices(self.both_aug_ops, k=self.cfg.DATA_AUGTIME-time_img)
		for op in ops:
			sample = op(sample)
		sample = self.imagenorm(sample)
		if self.cfg.DATA_RANDOMCROP > 0:
			sample = self.randomcrop(sample)
		sample['image_strong'] = sample['image']
		return sample

	def __image_strong_augment__(self, image):
		sample = {'image': Image.fromarray(image)}
		time_img = random.randint(0,2)
		ops = random.choices(self.img_aug_ops, k=time_img)
		for op in ops:
			sample = op(sample)
		sample['image'] = np.asarray(sample['image'])
		sample = self.imagenorm(sample)
		return sample['image']
		
	def __len__(self):
		raise NotImplementedError

	def load_name(self, idx):
		raise NotImplementedError	

	def load_image(self, idx):
		raise NotImplementedError	

	def load_segmentation(self, idx):
		raise NotImplementedError

	def load_pseudo_segmentation(self, idx):
		raise NotImplementedError

	def load_feature(self, idx):
		raise NotImplementedError
		
	def save_result(self, result_list, model_id):
		raise NotImplementedError	

	def save_pseudo_gt(self, result_list, level=None):
		raise NotImplementedError

	def do_python_eval(self, model_id):
		raise NotImplementedError
