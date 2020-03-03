import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
import argparse
import shutil
import tqdm
import setproctitle
import nibabel as nib
import SimpleITK as sitk
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.atrous_denseunet import ADenseUnet
from models.vnet_o import VNet
from dataset.abus import ABUS, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.utils import get_dice, load_config

def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./config/config.py')
    parser.add_argument('--gpu', default='2', type=str)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--fold', type=str, default='1')
    parser.add_argument('--arch', type=str, default='denseunet', 
                        choices=('denseunet', 'vnet'))

    parser.add_argument('--save')
    parser.add_argument('--resume', type=str, metavar='PATH')


    args = parser.parse_args()
    cfg = load_config(args.input)
    return cfg, args


def test(args, test_loader, net, save_result=True):
    net.eval()
    mean_dice = []
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
            dice = get_dice(pred, label)
            mean_dice.append(dice)

            if save_result:
                save_name = os.path.join(args.save, case_name[0][:-4])
                if not os.path.exists(save_name):
                    os.makedirs(save_name)

                img = sitk.GetImageFromArray(image)
                sitk.WriteImage(img, save_name + '/' + "img.nii.gz")

                img = sitk.GetImageFromArray(label)
                sitk.WriteImage(img, save_name + '/' + "gt.nii.gz")

                img = sitk.GetImageFromArray(pred)
                sitk.WriteImage(img, save_name + '/' + "pred.nii.gz")

        return np.mean(mean_dice)

def main():

    # --- init args ---
    cfg, args = get_args()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    setproctitle.setproctitle(args.save)


    # --- building network ---
    if args.arch == 'denseunet':
        net = ADenseUnet(in_channels=cfg.general.in_channels, 
                        num_classes=cfg.general.num_classes,
                        drop_rate=cfg.training.drop_rate)
    elif args.arch == 'vnet':
        net = VNet(n_channels=cfg.general.in_channels, 
                   n_classes=cfg.general.num_classes)
    else:
        raise(RuntimeError('No module named {}'.format(args.arch)))
    net = net.cuda()
    #model = nn.parallel.DataParallel(model, list(range(args.ngpu)))


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
    test_set = ABUS(base_dir=cfg.general.root_path,
                    split='test',
                    fold=args.fold,
                    transform=test_transform
                    )
    kwargs = {'num_workers': cfg.training.num_workers, 'pin_memory': cfg.training.pin_memory}
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)


    # --- testing ---
    dice = test(args, test_loader, net)
    print('average dice is {}'.format(dice))

if __name__ == "__main__":
    main()
