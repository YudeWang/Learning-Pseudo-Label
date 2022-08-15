# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

#import net.resnet_atrous as atrousnet
#import net.resnet as resnet
#import net.xception as xception
#import net.vgg as vgg
#import net.resnet38d as resnet38d
#import net.mobilenetv3 as mobilenetv3
#import net.mobilenetv2 as mobilenetv2
#import net.efficientnet as efficientnet
from utils.registry import BACKBONES

def build_backbone(backbone_name, pretrained=True, **kwargs):
	net = BACKBONES.get(backbone_name)(pretrained=pretrained, **kwargs)
	return net
