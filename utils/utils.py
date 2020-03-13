import os
import sys
import torch
import shutil
import logging
import importlib
from medpy import metric
import numpy as np

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

def get_metrics(pred, gt, voxelspacing=(0.5, 0.5, 0.5)):
    r""" 
    Get statistic metrics of segmentation

    These metrics include: Dice, Jaccard, Hausdorff Distance, 95% Hausdorff Distance, 
    and Average surface distance(ASD) metric.

    If the prediction result is 0s, we set hd, hd95, asd 10.0 to avoid errors.

    Parameters:
    -----------
    pred: 3D numpy ndarray
        binary prediction results 

    gt: 3D numpy ndarray
        binary ground truth

    voxelspacing: tuple of 3 floats. default: (0.5, 0.5, 0.5)
        voxel space of 3D image

    Returns:
    --------
    metrics: dict of 5 metrics 
        dict{dsc, jc, hd, hd95, asd}
    """

    dsc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    vs = np.sum(gt) * voxelspacing[0] * voxelspacing[1] * voxelspacing[2]

    if np.sum(pred) == 0:
        print('=> prediction is 0s! ')
        hd = 10.
        hd95 = 10.
        asd = 10.
    else:
        hd = metric.binary.hd(pred, gt, voxelspacing=voxelspacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)
        asd = metric.binary.asd(pred, gt, voxelspacing=voxelspacing)

    metrics = {'dsc': dsc, 'jc': jc, 'hd': hd, 'hd95': hd95, 'asd': asd, 'vs': vs} 
    return metrics 

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
