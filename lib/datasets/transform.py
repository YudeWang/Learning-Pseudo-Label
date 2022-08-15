# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import cv2
import torch
import random
import PIL
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):

		h, w = sample['image'].shape[:2]
		ch = min(h, self.output_size[0])
		cw = min(w, self.output_size[1])
		
		h_space = h - self.output_size[0]
		w_space = w - self.output_size[1]

		if w_space > 0:
			cont_left = 0
			img_left = random.randrange(w_space+1)
		else:
			cont_left = random.randrange(-w_space+1)
			img_left = 0

		if h_space > 0:
			cont_top = 0
			img_top = random.randrange(h_space+1)
		else:
			cont_top = random.randrange(-h_space+1)
			img_top = 0

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img_crop = np.zeros((self.output_size[0], self.output_size[1], 3), np.float32)
				img_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 img[img_top:img_top+ch, img_left:img_left+cw]
				#img_crop = img[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = img_crop
			elif 'segmentation' == key:
				seg = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 seg[img_top:img_top+ch, img_left:img_left+cw]
				#seg_crop = seg[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 seg_pseudo[img_top:img_top+ch, img_left:img_left+cw]
				#seg_crop = seg_pseudo[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop
			elif 'feature' in key:
				feature = sample[key]
				c,h,w = feature.shape
				feature_crop = np.zeros((c, self.output_size[0], self.output_size[1]), np.float32)
				feature_crop[:,cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 feature[:, img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = feature_crop
		return sample

class RandomHSV(object):
	"""Generate randomly the image in hsv space."""
	def __init__(self, h_r, s_r, v_r):
		self.h_r = h_r
		self.s_r = s_r
		self.v_r = v_r

	def __call__(self, sample):
		image = sample['image']
		hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		h = hsv[:,:,0].astype(np.int32)
		s = hsv[:,:,1].astype(np.int32)
		v = hsv[:,:,2].astype(np.int32)
		delta_h = random.randint(-self.h_r,self.h_r)
		delta_s = random.randint(-self.s_r,self.s_r)
		delta_v = random.randint(-self.v_r,self.v_r)
		h = (h + delta_h)%180
		s = s + delta_s
		s[s>255] = 255
		s[s<0] = 0
		v = v + delta_v
		v[v>255] = 255
		v[v<0] = 0
		hsv = np.stack([h,s,v], axis=-1).astype(np.uint8)	
		image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
		sample['image'] = image
		return sample

class RandomAutoContrast(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		alpha = random.uniform(self.range[0], self.range[1])
		img = sample['image']
		imgc = ImageOps.autocontrast(img)
		sample['image'] = Image.blend(imgc, img, alpha)
		return sample

class RandomBrightness(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		alpha = random.uniform(self.range[0], self.range[1])
		img = sample['image']
		sample['image'] = PIL.ImageEnhance.Brightness(img).enhance(alpha)
		return sample

class RandomColor(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		alpha = random.uniform(self.range[0], self.range[1])
		img = sample['image']
		sample['image'] = PIL.ImageEnhance.Color(img).enhance(alpha)
		return sample

class RandomContrast(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		alpha = random.uniform(self.range[0], self.range[1])
		img = sample['image']
		sample['image'] = PIL.ImageEnhance.Contrast(img).enhance(alpha)
		return sample

class RandomCutout(object):
	def __init__(self, alpha, mean):
		self.alpha = alpha
		self.mean = [int(mean[0]*255), int(mean[1]*255), int(mean[2]*255)] if mean is not None else [127,127,127]
	def __call__(self, sample):
		h = sample['row']
		w = sample['col']
		height = int(random.random()*self.alpha*h)
		width = int(random.random()*self.alpha*w)
		h0 = int(random.random()*(h-height))
		w0 = int(random.random()*(w-width))

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key].copy()
				img[h0:h0+height, w0:w0+width, :] = self.mean
				sample[key] = img 
			elif 'segmentation' == key:
				seg = sample[key].copy()
				seg[h0:h0+height, w0:w0+width] = 255
				sample[key] = seg
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_pseudo[h0:h0+height, w0:w0+width] = 255
				sample[key] = seg_pseudo
		return sample

class RandomEqualize(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		alpha = random.uniform(self.range[0], self.range[1])
		img = sample['image']
		imgc = ImageOps.equalize(img)
		sample['image'] = Image.blend(imgc, img, alpha)
		return sample

class RandomInvert(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		alpha = random.uniform(self.range[0], self.range[1])
		img = sample['image']
		img_ivt = ImageOps.invert(img)
		sample['image'] = Image.blend(img_ivt, img, alpha)
		return sample

class RandomPosterize(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		bit = random.randint(self.range[0], self.range[1])
		img = sample['image']
		sample['image'] = ImageOps.posterize(img, bit)
		return sample

class RandomSharpness(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		alpha = random.uniform(self.range[0], self.range[1])
		img = sample['image']
		sample['image'] = PIL.ImageEnhance.Sharpness(img).enhance(alpha)
		return sample

class RandomSolarize(object):
	def __init__(self, r):
		self.range = r
	def __call__(self, sample):
		alpha = random.uniform(self.range[0], self.range[1])*255
		img = sample['image']
		sample['image'] = PIL.ImageOps.solarize(img, alpha)
		return sample

class RandomShear(object):
	"""Randomly rotate image"""
	def __init__(self, rx, ry, mean, is_continuous=False):
		self.rangex = rx
		self.rangey = ry
		self.mean = [int(mean[0]*255), int(mean[1]*255), int(mean[2]*255)] if mean is not None else [127,127,127]
		self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

	def __call__(self, sample):
		row, col, _ = sample['image'].shape
		alphax = random.uniform(self.rangex[0], self.rangex[1])
		alphay = random.uniform(self.rangey[0], self.rangey[1])
		m = np.float32([[1, alphax, 0], [alphay, 1, 0]])
		m[0,2] = -m[0,1] * col/2
		m[1,2] = -m[1,0] * row/2
		
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = cv2.warpAffine(img, m, (col,row), flags=cv2.INTER_CUBIC, borderValue=self.mean)
				sample[key] = img 
			elif 'segmentation' == key:
				seg = sample[key]
				seg = cv2.warpAffine(seg, m, (col,row), flags=self.seg_interpolation, borderValue=255)
				sample[key] = seg
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_pseudo = cv2.warpAffine(seg_pseudo, m, (col,row), flags=self.seg_interpolation, borderValue=255)
				sample[key] = seg_pseudo
		return sample

class RandomFlip(object):
	"""Randomly flip image"""
	def __init__(self, threshold):
		self.flip_t = threshold
	def __call__(self, sample):
		if random.random() < self.flip_t:
			key_list = sample.keys()
			for key in key_list:
				if 'image' in key:
					img = sample[key]
					img = np.flip(img, axis=1)
					sample[key] = img
				elif 'segmentation' == key:
					seg = sample[key]
					seg = np.flip(seg, axis=1)
					sample[key] = seg
				elif 'segmentation_pseudo' in key:
					seg_pseudo = sample[key]
					seg_pseudo = np.flip(seg_pseudo, axis=1)
					sample[key] = seg_pseudo
				elif 'feature' in key:
					feature = sample[key]
					feature = np.flip(feature, axis=2)
					sample[key] = feature
		return sample

class RandomRotation(object):
	"""Randomly rotate image"""
	def __init__(self, angle_r, mean, is_continuous=False):
		self.angle_r = angle_r
		self.mean = [int(mean[0]*255), int(mean[1]*255), int(mean[2]*255)] if mean is not None else [127,127,127]
		self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

	def __call__(self, sample):
		row, col, _ = sample['image'].shape
		rand_angle = random.randint(self.angle_r[0], self.angle_r[1])
		m = cv2.getRotationMatrix2D(center=(col/2, row/2), angle=rand_angle, scale=1)
		
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = cv2.warpAffine(img, m, (col,row), flags=cv2.INTER_CUBIC, borderValue=self.mean)
				sample[key] = img 
			elif 'segmentation' == key:
				seg = sample[key]
				seg = cv2.warpAffine(seg, m, (col,row), flags=self.seg_interpolation, borderValue=255)
				sample[key] = seg
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_pseudo = cv2.warpAffine(seg_pseudo, m, (col,row), flags=self.seg_interpolation, borderValue=255)
				sample[key] = seg_pseudo
		return sample

class RandomGaussian(object):
	def __init__(self, threshold):
		self.threshold = threshold
	
	def __call__(self, sample):
		r = random.random() 
		key_list = sample.keys()
		if random.random() < self.threshold:
			img = Image.fromarray(sample['image'])
			img = img.filter(ImageFilter.GaussianBlur(radius=r))
			img = np.array(img)
			#img = cv2.GaussianBlur(img,(r,r),0)
			sample['image'] = img
		return sample

class RandomScale(object):
	"""Randomly scale image"""
	def __init__(self, scale_r, is_continuous=False):
		self.scale_r = scale_r
		self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

	def __call__(self, sample):
		row, col, _ = sample['image'].shape
		rand_scale = random.random()*(self.scale_r[1] - self.scale_r[0]) + self.scale_r[0]
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = cv2.resize(img, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
				sample[key] = img
			elif 'segmentation' == key:
				seg = sample[key]
				seg = cv2.resize(seg, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
				sample[key] = seg
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_pseudo = cv2.resize(seg_pseudo, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
				sample[key] = seg_pseudo
			elif 'feature' in key:
				feature = sample[key]
				feature = feature.transpose((1,2,0))
				feature = cv2.resize(feature, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
				feature = feature.transpose((2,0,1))
				sample[key] = feature
		return sample

class RandomCopyPaste(object):
	"""copy object by mask into another image"""
	"""sample1 provide background"""
	def __init__(self, num=1):
		self.num = num
		
	def __call__(self, sample1, sample2, t='labeled'):
		assert 'segmentation_pseudo' not in sample1.keys()
		assert t in ['labeled', 'unlabeled']
		img1,seg1,cls1 = sample1['image'],sample1['segmentation'],sample1['category']
		img2,seg2,cls2 = sample2['image'],sample2['segmentation'],sample2['category']
		h,w,_ = img1.shape
		cls_idx1 = []
		cls_idx2 = []
		for i in range(1,cls1.shape[0]):
			if cls1[i] > 0:
				cls_idx1.append(i)
			if cls2[i] > 0:
				if np.sum(seg2==i) >= 640:
					cls_idx2.append(i)
		if len(cls_idx2) == 0:
			return sample1
		np.random.shuffle(cls_idx1)
		np.random.shuffle(cls_idx2)
			
		if t == 'labeled':
			cutin_idx = random.randint(0, len(cls_idx1))
		elif t == 'unlabeled':
			cutin_idx = len(cls_idx1)
			
		img = img1.copy()
		seg = seg1.copy()
		cls = cls1.copy()
		for i in range(len(cls_idx1)+1):
			if i < cutin_idx:
				continue	
			elif i == cutin_idx:
			#	left = random.randrange(-w//2, w//2)
			#	top = random.randrange(-h//2, h//2)

			#	left1 = max(left,0)
			#	right1 = min(w,w+left)
			#	top1 = max(top,0)
			#	bottom1 = min(h,h+top)

			#	left2 = max(-left,0)
			#	right2 = min(w,w-left)
			#	top2 = max(-top,0)
			#	bottom2 = min(h,h-top)
			#	#img[seg2 == cls_idx2[0]] = img2[seg2 == cls_idx2[0]]
			#	#seg[seg2 == cls_idx2[0]] = seg2[seg2 == cls_idx2[0]]

			#	subimg = img[top1:bottom1, left1:right1, :]
			#	subseg = seg[top1:bottom1, left1:right1]
			#	subimg2 = img2[top2:bottom2, left2:right2, :]
			#	subseg2 = seg2[top2:bottom2, left2:right2]
			#	subimg[subseg2==cls_idx2[0]] = subimg2[subseg2==cls_idx2[0]]
			#	subseg[subseg2==cls_idx2[0]] = subseg2[subseg2==cls_idx2[0]]

			#	##################
				img[seg2 == cls_idx2[0]] = img2[seg2 == cls_idx2[0]]
				seg[seg2 == cls_idx2[0]] = seg2[seg2 == cls_idx2[0]]
				
			else:
				img[seg1 == cls_idx1[i-1]] = img1[seg1 == cls_idx1[i-1]]
				seg[seg1 == cls_idx1[i-1]] = seg1[seg1 == cls_idx1[i-1]]
		if cls1[cls_idx2[0]] == 0:
			sample1['category_copypaste'][cls_idx2[0]] = 1
		cls[cls_idx2[0]] = 1

		sample1['image'] = img
		sample1['segmentation'] = seg
		sample1['category'] = cls
		return sample1

			
			

class ImageNorm(object):
	"""Randomly scale image"""
	def __init__(self, mean=None, std=None):
		self.mean = mean
		self.std = std
	def __call__(self, sample):
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				image = sample[key].astype(np.float32)
				if self.mean is not None and self.std is not None:
					image[...,0] = (image[...,0]/255 - self.mean[0]) / self.std[0]
					image[...,1] = (image[...,1]/255 - self.mean[1]) / self.std[1]
					image[...,2] = (image[...,2]/255 - self.mean[2]) / self.std[2]
				else:
					image /= 255.0
				sample[key] = image
		return sample

class Multiscale(object):
	def __init__(self, rate_list):
		self.rate_list = rate_list

	def __call__(self, sample):
		image = sample['image']
		row, col, _ = image.shape
		image_multiscale = []
		for rate in self.rate_list:
			rescaled_image = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
			sample['image_%f'%rate] = rescaled_image
		return sample


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				image = sample[key].astype(np.float32)
				# swap color axis because
				# numpy image: H x W x C
				# torch image: C X H X W
				image = image.transpose((2,0,1))
				sample[key] = torch.from_numpy(image)
				#sample[key] = torch.from_numpy(image.astype(np.float32)/128.0-1.0)
			elif 'edge' == key:
				edge = sample['edge']
				sample['edge'] = torch.from_numpy(edge.astype(np.float32))
				sample['edge'] = torch.unsqueeze(sample['edge'],0)
			elif 'segmentation' == key:
				segmentation = sample['segmentation']
				sample['segmentation'] = torch.from_numpy(segmentation.astype(np.long)).long()
			elif 'segmentation_pseudo' in key:
				segmentation_pseudo = sample[key]
				sample[key] = torch.from_numpy(segmentation_pseudo.astype(np.float32)).long()
			elif 'segmentation_onehot' == key:
				onehot = sample['segmentation_onehot'].transpose((2,0,1))
				sample['segmentation_onehot'] = torch.from_numpy(onehot.astype(np.float32)).float()
			elif 'category' in key:
				sample[key] = torch.from_numpy(sample[key].astype(np.float32)).float()
			elif 'mask' == key:
				mask = sample['mask']
				sample['mask'] = torch.from_numpy(mask.astype(np.float32))
			elif 'feature' == key:
				feature = sample['feature']
				sample['feature'] = torch.from_numpy(feature.astype(np.float32))
		return sample

