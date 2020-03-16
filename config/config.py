from easydict import EasyDict
import numpy as np

cfg = EasyDict() 

#######################
#      general        #
#######################
cfg.general = {}
cfg.general.in_channels = 1
cfg.general.num_classes = 2
cfg.general.crop_size = [128, 64, 128]

#######################
# training parameters #
#######################
cfg.training = {}
# optimizer
cfg.training.opt = 'adam'
cfg.training.weight_decay = 1e-4
cfg.training.lr = 1e-3
cfg.training.drop_rate = 0.0
# random flag 
cfg.training.deterministic = True 
cfg.training.seed = 2020
# dataset
cfg.training.num_workers = 4
cfg.training.pin_memory = True
