import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2' 
import argparse
import shutil
import tqdm
import setproctitle
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from models.atrous_denseunet import ADenseUnet
from models.vnet_o import VNet
from unet import UNet3D
from dataset.abus import ABUS
from dataset.augment import ToTensor
from utils.utils import load_config, get_metrics

def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./config/config.py')
    # path
    parser.add_argument('--root_path', type=str, default='/data/xuyangcao/code/data/roi_3d/abus_shift')
    # gpu
    parser.add_argument('--ngpu', type=int, default=1)
    # evaluate
    #parser.add_argument('-e', '--evaluate', action='store_true', default=False)

    # frequently changed params
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--arch', type=str, default='denseunet', choices=('denseunet', 'vnet', 'unet3d', 'resunet3d'))
    parser.add_argument('--save', default=None, type=str) 
    parser.add_argument('--resume', type=str)
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()
    cfg = load_config(args.input)
    return cfg, args


def test(args, test_loader, net):
    net.eval()

    dsc_list = []
    jc_list = []
    hd_list = []
    hd95_list = []
    asd_list = []
    precision_list = []
    recall_list = []
    vs_list = []
    filename_list = []
    with torch.no_grad():
        for sample in tqdm.tqdm(test_loader):
            image, label = sample['image'], sample['label']
            case_name = sample['filename']

            image = image.cuda()
            out = net(image)
            pred = out.max(1)[1]
            
            image = image[0][0].cpu().numpy()
            image = (image + 0.5) * 0.5
            image = image.astype(np.float)

            label = label[0].cpu().numpy()
            label = label.astype(np.float)
            pred = pred[0].cpu().numpy()
            pred = pred.astype(np.float)

            # get metrics
            metrics = get_metrics(pred, label, voxelspacing=(0.5, 0.5, 0.5)) 
            dsc_list.append(metrics['dsc'])
            jc_list.append(metrics['jc'])
            hd_list.append(metrics['hd'])
            hd95_list.append(metrics['hd95'])
            asd_list.append(metrics['asd'])
            precision_list.append(metrics['precision'])
            recall_list.append(metrics['recall'])
            vs_list.append(metrics['vs'])
            filename_list.append(case_name)


            if args.save_image:
                save_name = os.path.join(args.save_path, case_name[0][:-4])
                if not os.path.exists(save_name):
                    os.makedirs(save_name)

                img = sitk.GetImageFromArray(image)
                sitk.WriteImage(img, save_name + '/' + "img.nii.gz")

                img = sitk.GetImageFromArray(label)
                sitk.WriteImage(img, save_name + '/' + "gt.nii.gz")

                img = sitk.GetImageFromArray(pred)
                sitk.WriteImage(img, save_name + '/' + "pred.nii.gz")

        df = pd.DataFrame()
        df['filename'] = filename_list 
        df['dsc'] = np.array(dsc_list)
        df['jc'] = np.array(jc_list) 
        df['hd95'] = np.array(hd95_list) 
        df['precision'] = np.array(precision_list)
        df['recall'] = np.array(recall_list)
        print(df.describe())
        df['hd'] = np.array(hd_list) 
        df['asd'] = np.array(asd_list) 
        df['volume(mm^3)'] = np.array(vs_list)
        df.to_excel(args.csv_file_name)


def main():

    # --- init args ---
    cfg, args = get_args()
    
    # --- saving path ---
    if 'best' in args.resume:
        file_name = 'model_best_result'
    elif 'check' in args.resume:
        file_name = 'checkpoint_result'
    else:
        raise(RuntimeError('Error in args.resume'))
    csv_file_name = file_name + '.xlsx'

    if args.save is not None:
        save_path = os.path.join(args.save, file_name) 
        csv_path = os.save
    else:
        save_path = os.path.join(os.path.dirname(args.resume), file_name)
        csv_path = os.path.dirname(args.resume)
    args.save_path = save_path
    args.csv_file_name = os.path.join(csv_path, csv_file_name) 
    if args.save_image:
        print('=> saving images in :', args.save_path)
    else:
        print('=> we did not save segmentation results!')
    print('=> saving csv in :', args.csv_file_name)


    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    setproctitle.setproctitle(args.save_path)


    # --- building network ---
    if args.arch == 'denseunet':
        net = ADenseUnet(in_channels=cfg.general.in_channels, 
                        num_classes=cfg.general.num_classes,
                        drop_rate=cfg.training.drop_rate)
    elif args.arch == 'unet3d':
        net = UNet3D(residual=False)
    elif args.arch == 'resunet3d':
        net = UNet3D(residual=True)
    elif args.arch == 'vnet':
        net = VNet(n_channels=cfg.general.in_channels, 
                   n_classes=cfg.general.num_classes)
    else:
        raise(RuntimeError('No module named {}'.format(args.arch)))
    net = net.cuda()
    if args.arch == 'unet3d' or args.arch == 'resunet3d':
        net = nn.parallel.DataParallel(net, list(range(args.ngpu)))


    # --- resume trained weights ---
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_pre']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        raise(RuntimeError('resume is None!'))


    # --- preparing dataset
    test_transform = transforms.Compose([ToTensor()])
    test_set = ABUS(base_dir=args.root_path,
                    split='test',
                    fold=args.fold,
                    transform=test_transform
                    )
    kwargs = {'num_workers': cfg.training.num_workers, 'pin_memory': cfg.training.pin_memory}
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)


    # --- testing ---
    test(args, test_loader, net)

if __name__ == "__main__":
    main()
