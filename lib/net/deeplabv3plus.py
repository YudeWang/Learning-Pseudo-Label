# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net.backbone import build_backbone
from net.operators import ASPP
from utils.registry import NETS

@NETS.register_module
class deeplabv3plus(nn.Module):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3plus, self).__init__()
		self.cfg = cfg
		self.batchnorm = batchnorm
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=cfg.MODEL_BACKBONE_PRETRAIN, norm_layer=self.batchnorm, **kwargs)
		input_channel = self.backbone.OUTPUT_DIM
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=[0, 6, 12, 18],
				bn_mom = cfg.TRAIN_BN_MOM,
				has_global = cfg.MODEL_ASPP_HASGLOBAL,
				batchnorm = self.batchnorm)

		indim = self.backbone.MIDDLE_DIM
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, 3, 1, padding=1, bias=False),
				batchnorm(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
				batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
				batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
		)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		for m in self.modules():
			if m not in self.backbone.modules():
		#		if isinstance(m, nn.Conv2d):
		#			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if isinstance(m, batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		if cfg.MODEL_FREEZEBN:
			self.freeze_bn()

	def forward(self, x, getf=False, interpolate=True, mid=False):
		N,C,H,W = x.size()
		l1, l2, l3, l4 = self.backbone(x)
		feature_aspp = self.aspp(l4)

		feature_shallow = self.shortcut_conv(l1)
		n,c,h,w = feature_shallow.size()
		feature_aspp = F.interpolate(feature_aspp,(h,w),mode='bilinear',align_corners=True)

		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		feature = self.cat_conv(feature_cat) 
		result = self.cls_conv(feature)
		result = F.interpolate(result, (H,W), mode='bilinear',align_corners=True)


		if getf:
			if interpolate:
				feature = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
			if mid:
				return result, feature, l4
			return result, feature
		else:
			if mid:
				return result, l4
			else:
				return result


