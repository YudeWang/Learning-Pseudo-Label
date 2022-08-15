# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import os
import shutil
from config import config_dict as cfg_dict1
from config_retrain import config_dict as cfg_dict2
from train import train_net as train
from test import test_net as test
from inference import inference
from retrain import train_net as retrain
from retest import test_net as retest
from utils.configuration import Configuration

if __name__ == '__main__':
    cfg1 = Configuration(cfg_dict1)
    cfg2 = Configuration(cfg_dict2)
    cfg1.MODEL_BACKBONE = 'resnet101'
    cfg2.MODEL_BACKBONE = 'resnet101'
    subsize = [92,183,366,732,1464]
    for s in subsize:
        # Stage1 - init
        shutil.rmtree(cfg1.LOG_DIR)
        os.mkdir(cfg1.LOG_DIR)
        comment = f'sp{cfg1.TRAIN_SCORE_POW}'
        print(comment)
        pseudopath = '../../data/VOCdevkit/pseudo_gt/res101-ours/%s'%comment
        cfg1.DATA_SPLIT = s 
        cfg1.DATA_PSEUDO_GT = None

        # Stage1 - train
        train(cfg1,comment)

        # Stage1 - test
        checkpoint = f'{cfg1.MODEL_NAME}_{cfg1.MODEL_BACKBONE}_' \
                     f'{cfg1.DATA_NAME}_itr{cfg1.TRAIN_ITERATION}_' \
                     f'sub{cfg1.DATA_SPLIT}_{comment}_hybridnet.pth'
        cfg1.TEST_CKPT = os.path.join(cfg1.MODEL_SAVE_DIR, checkpoint)
        test(cfg1)

        # Stage1 - inference
        inference(cfg1,pseudopath)

        #################################################

        # Stage2 - init
        shutil.rmtree(cfg2.LOG_DIR)
        os.mkdir(cfg2.LOG_DIR)
        cfg2.DATA_SPLIT = s 
        cfg2.DATA_PSEUDO_GT = pseudopath

        # Stage2 - train
        retrain(cfg2,comment)

        # Stage2 - test
        checkpoint_retrain = f'{cfg2.MODEL_NAME}_{cfg2.MODEL_BACKBONE}_' \
                             f'{cfg2.DATA_NAME}_itr{cfg2.TRAIN_ITERATION}_' \
                             f'sub{cfg2.DATA_SPLIT}_{comment}_retrain.pth'
        cfg2.TEST_CKPT = os.path.join(cfg2.MODEL_SAVE_DIR, checkpoint_retrain)
        retest(cfg2)


