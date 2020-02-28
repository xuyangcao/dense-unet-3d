from easydict import EasyDict
import numpy as np

cfg = EasyDict() 


#######################
#        paths        #
#######################
cfg.path = {}
# datset root path 
cfg.path.root_path = '/data/xuyangcao/code/data/roi_3d/abus_roi'
# path to save model
cfg.path.save = './work/test-1'
# path to save tensorboard log
cfg.path.log_dir = './log/abus_roi'


#######################
#      general        #
#######################
cfg.general = {}
cfg.general.in_channels = 1
cfg.general.num_classes = 2
cfg.general.crop_size = [64, 128, 128]
# architecture
cfg.general.arch = 'denseunet' # choices = ['a-denseunet', 'vnet', 'unet-3d']


#######################
# training parameters #
#######################
cfg.training = {}
# epoch 
cfg.training.start_epoch = 1
cfg.training.n_epochs = 300
# batch size 
cfg.training.batch_size = 2
# gpu
cfg.training.ngpu = 1
cfg.training.gpu = '1'
# cross validation fold number
cfg.training.fold = '1'
# optimizer
cfg.training.opt = 'adam'
cfg.training.weight_decay = 1e-8
cfg.training.lr = 1e-4
# random flag 
cfg.training.deterministic = True 
cfg.training.seed = 2020
# dataset
cfg.training.num_workers = 4
cfg.training.pin_memory = True


#######################
# testing parameters  #
#######################
cfg.testing = {}
cfg.testing.fold = '1'
cfg.testing.batch_size = 1
