from easydict import EasyDict
import numpy as np

cfg = EasyDict() 

#######################
#      general        #
#######################
cfg.general = {}
cfg.general.in_channels = 1
cfg.general.num_classes = 2
cfg.general.crop_size = [64, 128, 128]
# datset root path 
cfg.general.root_path = '/data/xuyangcao/code/data/roi_3d/abus_roi'
# path to save tensorboard log
cfg.general.log_dir = './log/abus_roi'


#######################
# training parameters #
#######################
cfg.training = {}
# epoch 
cfg.training.start_epoch = 1
cfg.training.n_epochs = 300
# optimizer
cfg.training.opt = 'adam'
cfg.training.weight_decay = 1e-8
cfg.training.lr = 1e-4
cfg.training.drop_rate = 0.3
# random flag 
cfg.training.deterministic = True 
cfg.training.seed = 2020
# dataset
cfg.training.num_workers = 4
cfg.training.pin_memory = True
