# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

config_dict = {
		'EXP_NAME': 'deeplabv3+_voc_swsss',
		'GPUS': 1,

		'DATA_NAME': 'SemiWeak_VOCDataset',
		'DATA_YEAR': 2012,
		'DATA_AUG': True,
		'DATA_WORKERS': 4,
		'DATA_MEAN': [0.485, 0.456, 0.406],
		'DATA_STD': [0.229, 0.224, 0.225],
		'DATA_AUGTIME': 2,
		'DATA_RANDOMCROP': 384,
		'DATA_RANDOMSCALE': [0.5, 2.0],
		'DATA_RANDOM_H': 10,
		'DATA_RANDOM_S': 10,
		'DATA_RANDOM_V': 10,
		'DATA_RANDOMFLIP': 0.5,
		'DATA_RANDOMGAUSSIAN': 0.5,
		'DATA_RANDOMMIXUP': False,
		'DATA_RANDOMAUTOCONTRAST': [0,1],
		'DATA_RANDOMBRIGHTNESS': [0.5,1.5],
		'DATA_RANDOMCOLOR': [0.5,1.5],
		'DATA_RANDOMCONTRAST': [0.5,1.5],
		'DATA_RANDOMEQUALIZE': [0.5,1.5],
		'DATA_RANDOMINVERT': [0,1],
		'DATA_RANDOMPOSTERIZE': [1,8],
		'DATA_RANDOMSHARPNESS': [0,1],
		'DATA_RANDOMSOLARIZE': [0,1],
		'DATA_RANDOMSHEARX': [-0.3, 0.3],
		'DATA_RANDOMSHEARY': [-0.3, 0.3],
		'DATA_RANDOMROTATION': [-45, 45],
		'DATA_RANDOMCUTOUT': None,
		'DATA_RANDOMCOPYPASTE': 0,
		'DATA_PSEUDO_GT': '../../data/VOCdevkit/pseudo_gt/res101/sub92',
		'DATA_SPLIT': 92,
		
		'MODEL_NAME': 'deeplabv3plus',
		'MODEL_BACKBONE': 'resnet101',
		'MODEL_BACKBONE_PRETRAIN': True,
		'MODEL_BACKBONE_DILATED': True,
		'MODEL_BACKBONE_MULTIGRID': False,
		'MODEL_BACKBONE_DEEPBASE': True,
		'MODEL_SHORTCUT_DIM': 48,
		'MODEL_OUTPUT_STRIDE': 8,
		'MODEL_ASPP_OUTDIM': 256,
		'MODEL_ASPP_HASGLOBAL': True,
		'MODEL_TRANSFORMER_ENCODER_NUM': 1,
		'MODEL_NUM_CLASSES': 21,
		'MODEL_FREEZEBN': False,

		'TRAIN_LR': 0.007,
		'TRAIN_LR_GAMMA': 0.1,
		'TRAIN_MOMENTUM': 0.9,
		'TRAIN_WEIGHT_DECAY': 4e-5,
		'TRAIN_BN_MOM': 0.1,
		'TRAIN_POWER': 0.9,
		'TRAIN_BATCHES': 16,
		'TRAIN_SHUFFLE': True,
		'TRAIN_MINEPOCH': 0,
		'TRAIN_EPOCHS': 300,
		'TRAIN_ITERATION': 30000,
		'TRAIN_LOSS_LAMBDA': 0,
		'TRAIN_AFN_LAMBDA': 0.01,
		'TRAIN_AFN_DELTA': 0.3,
		'TRAIN_UDA_FACTOR': 0.1,
		'TRAIN_UDA_TEMP': 0.8,
		'TRAIN_TBLOG': True,

		'TEST_MULTISCALE': [0.5, 0.75, 1.0, 1.25, 1.5],
		'TEST_FLIP': True,
		'TEST_CRF': True,
		'TEST_BATCHES': 1,		
}

config_dict['ROOT_DIR'] = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
config_dict['MODEL_SAVE_DIR'] = os.path.join(config_dict['ROOT_DIR'],'model',config_dict['EXP_NAME'])
config_dict['TRAIN_CKPT'] = None
config_dict['LOG_DIR'] = os.path.join(config_dict['ROOT_DIR'],'log',config_dict['EXP_NAME'])
config_dict['TEST_CKPT'] = None

sys.path.insert(0, os.path.join(config_dict['ROOT_DIR'], 'lib'))
