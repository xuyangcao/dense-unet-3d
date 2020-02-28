import os
import sys
import torch
import shutil
import logging
import importlib
from medpy import metric

def save_checkpoint(state, is_best, path, prefix, filename='checkpoing.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

def get_metrics(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, hd95, asd

def get_dice(pred, gt):
    dice = metric.binary.dc(pred, gt)

    return dice


def load_config(file_name):
    r"""
    load configuration file as a python module

    Args:
        file_name: configuration file name

    Returns:
        a loaded network module            
    """

    dir_name = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    module_name, _ = os.path.splitext(base_name)
    #print('dir_name: ', dir_name)
    #print('base_name: ', base_name)
    #print('module_name: ', module_name)

    os.sys.path.append(dir_name)
    config = importlib.import_module(module_name)
    if module_name in sys.modules:
        importlib.reload(config)
    del os.sys.path[-1]

    return config.cfg


if __name__ == "__main__":
    cfg = load_config('../config/config.py')
    print(cfg)
