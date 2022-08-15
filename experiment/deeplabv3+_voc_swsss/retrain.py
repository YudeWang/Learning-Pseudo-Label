# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import os
import sys
import time
import copy
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from config import config_dict
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.imutils import img_denorm
from utils.visualization import generate_vis
from net.generateNet import generate_net
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasets.generateData import generate_dataset
from tensorboardX import SummaryWriter
torch.manual_seed(1) # cpu
torch.cuda.manual_seed_all(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def train_net(cfg, comment=''):
    period = 'train'
    transform = 'weak' if cfg.DATA_AUGTIME<=0 else 'strong'
    dataset = generate_dataset(cfg, period=period, transform=transform)
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    dataloader = DataLoader(dataset, 
                batch_size=cfg.TRAIN_BATCHES, 
                shuffle=False,
                num_workers=cfg.DATA_WORKERS,
                drop_last=True,
                worker_init_fn=worker_init_fn)
    
    net = generate_net(cfg, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
    if cfg.TRAIN_CKPT:
        net.load_state_dict(torch.load(cfg.TRAIN_CKPT),strict=False)
    if cfg.TRAIN_TBLOG:
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(cfg.LOG_DIR)    

    device = torch.device(0)
    if cfg.GPUS > 1:
        net = nn.DataParallel(net)
        parameter_source_teacher = net.module
    else:
        parameter_source_teacher = net
        
    net.to(device)        
    lr_rate = [1,1,10]
    net_optimizer = optim.SGD(
        params = [
            {'params': get_params(parameter_source_teacher, key='backbone'), 'lr': lr_rate[0]*cfg.TRAIN_LR},
            {'params': get_params(parameter_source_teacher, key='others'),   'lr': lr_rate[1]*cfg.TRAIN_LR},
            {'params': get_params(parameter_source_teacher, key='cls'),      'lr': lr_rate[2]*cfg.TRAIN_LR},
        ],
        momentum=cfg.TRAIN_MOMENTUM,
        weight_decay=cfg.TRAIN_WEIGHT_DECAY
    )
    itr = cfg.TRAIN_MINEPOCH * len(dataset)//cfg.TRAIN_BATCHES
    max_itr = cfg.TRAIN_ITERATION
    max_epoch = max_itr*(cfg.TRAIN_BATCHES)//len(dataset)+1
    tblogger = SummaryWriter(cfg.LOG_DIR)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.cuda.amp.GradScaler()
    with tqdm(total=max_itr) as pbar:
        for epoch in range(cfg.TRAIN_MINEPOCH, max_epoch):
            for i_batch, sample in enumerate(dataloader):
                now_lr = adjust_lr(net_optimizer, itr, max_itr, cfg.TRAIN_LR, cfg.TRAIN_POWER, lr_rate)
                net_optimizer.zero_grad()

                image = sample['image'].to(0)
                seg_label = sample['segmentation'].to(0)
                n,c,h,w = image.size()
                
                with torch.cuda.amp.autocast():
                    seg = net(image)
                    loss = criterion(seg, seg_label)

                scaler.scale(loss).backward()
                scaler.step(net_optimizer)
                scaler.update()

                pbar.set_description("%s, loss-%.3f" % (comment,loss.item()))
                pbar.update(1)
                time.sleep(0.001)

                if cfg.TRAIN_TBLOG and itr%100 == 0:
                    image_color = img_denorm(image[0].cpu().numpy()).astype(np.uint8)

                    seg_label_vis = seg_label[0].cpu().numpy()
                    label_color = dataset.label2colormap(seg_label_vis).transpose((2,0,1))

                    pred_color = vis_pred(seg[0], dataset.label2colormap)

                    tblogger.add_scalar('loss', loss.item(), itr)
                    tblogger.add_scalar('lr', now_lr, itr)
                    tblogger.add_image('Input', image_color, itr)
                    tblogger.add_image('Label', label_color, itr)
                    tblogger.add_image('Seg', pred_color, itr)
                itr += 1
                if itr>=max_itr:
                    break
            filename_prototype = f'{cfg.MODEL_NAME}_{cfg.MODEL_BACKBONE}_' \
                                 f'{cfg.DATA_NAME}_epoch{epoch}_{comment}'
            save_path = os.path.join(cfg.MODEL_SAVE_DIR,f'{filename_prototype}_retrain.pth')
            torch.save(parameter_source_teacher.state_dict(), save_path)
    filename_prototype = f'{cfg.MODEL_NAME}_{cfg.MODEL_BACKBONE}_' \
                         f'{cfg.DATA_NAME}_itr{cfg.TRAIN_ITERATION}_' \
                         f'sub{cfg.DATA_SPLIT}_{comment}'
    save_path = os.path.join(cfg.MODEL_SAVE_DIR,f'{filename_prototype}_retrain.pth')
    torch.save(parameter_source_teacher.state_dict(),save_path)
    print('%s has been saved'%save_path)
    if cfg.TRAIN_TBLOG:
        tblogger.close()
    writelog(cfg, period)

def adjust_lr(optimizer, itr, max_itr, lr_init, power, lr_rate):
    now_lr = lr_init * ((1 - itr/(max_itr+1e-5)) ** power) 
    optimizer.param_groups[0]['lr'] = now_lr * lr_rate[0]
    optimizer.param_groups[1]['lr'] = now_lr * lr_rate[1]
    optimizer.param_groups[2]['lr'] = now_lr * lr_rate[2]
    return now_lr

def get_params(model, key):
    for m in model.named_modules():
        if key == 'backbone':
            #if ('backbone' in m[0]) and isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d)):
            if ('backbone' in m[0]) and isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                for p in m[1].parameters():
                    yield p
        elif key == 'cls':
            if ('cls_conv' in m[0]) and isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                for p in m[1].parameters():
                    yield p
        elif key == 'others':
            if ('backbone' not in m[0] and 'cls_conv' not in m[0]) and isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                for p in m[1].parameters():
                    yield p

def vis_pred(pred,colorfunc):
    vis = torch.argmax(pred, dim=0)
    vis_color = colorfunc(vis.cpu().numpy()).transpose((2,0,1))
    return vis_color

if __name__ == '__main__':
    cfg = Configuration(config_dict)
    train_net(cfg)
