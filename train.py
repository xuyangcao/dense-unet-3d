import os
import sys
import random
import shutil
import logging
import argparse
import setproctitle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from models.atrous_denseunet import ADenseUnet
from models.vnet import VNet
from dataset.abus import ABUS, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.file_io import load_config
from utils.losses import dice_loss


def train(cfg, epoch, net, train_loader, optimizer, loss_fn, writer):
    net.train()
    batch_size = cfg.training.ngpu * cfg.training.batch_size

    loss_list = []
    dice_loss_list = []
    num_processed = 0
    num_train = len(train_loader.dataset)
    for batch_idx, sample in enumerate(train_loader):
        data, label = sample['image'], sample['label']
        data, label = data.cuda(), label.cuda()

        outputs = net(data)
        outputs_soft = F.softmax(outputs, dim=1)

        dice_loss = loss_fn['dice_loss'](outputs_soft[:, 1, ...], label == 1)
        loss = dice_loss

        # save losses to list
        dice_loss_list.append(dice_loss.item())
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        num_processed += len(data)
        partical_epoch = epoch + batch_idx / len(train_loader)
        logging.info('training epoch: {:.2f}/{} [{}/{} ({:.0f}%)\t loss: {:.8f}]'.format(
            partical_epoch, cfg.training.n_epochs, num_processed, num_train, 100. * batch_idx / len(train_loader), loss.item()))

        # show images on tensorboard
        with torch.no_grad():
            if batch_idx % 10 == 0:
                nrow = 5
                image = data[0, 0:1, :, 30:71:10, :].permute(2,0,1,3)
                image = (image + 0.5) * 0.5
                grid_image = make_grid(image, nrow=nrow)
                grid_image = grid_image.cpu().detach().numpy().transpose((1,2,0))

                gt = label[0, :, 30:71:10, :].unsqueeze(0).permute(2,0,1,3)
                grid_gt = make_grid(gt, nrow=nrow, normalize=False)
                grid_gt = grid_gt.cpu().detach().numpy().transpose((1,2,0))

                #outputs_soft = F.softmax(outputs, 1) #batchsize x num_classes x w x h x d
                prob = outputs_soft[0, 1:2, :, 30:71:10, :].permute(2,0,1,3)
                grid_prob = make_grid(prob, nrow=nrow, normalize=False)
                grid_prob = grid_prob.cpu().detach().numpy().transpose((1,2,0))

                fig = plt.figure()
                ax = fig.add_subplot(311)
                cs = ax.imshow(grid_image[:, :, 0], 'gray')
                fig.colorbar(cs, ax=ax, shrink=0.9)
                ax = fig.add_subplot(312)
                cs = ax.imshow(grid_gt[:, :, 0], 'hot', vmin=0., vmax=1.)
                fig.colorbar(cs, ax=ax, shrink=0.9)
                ax = fig.add_subplot(313)
                cs = ax.imshow(grid_prob[:, :, 0], 'hot', vmin=0, vmax=1.)
                fig.colorbar(cs, ax=ax, shrink=0.9)
                writer.add_figure('train/prediction_results', fig, epoch)
                fig.clear()

    # show scalars on tensorboard
    writer.add_scalar('train/dice_loss', float(np.mean(dice_loss_list)), epoch)
    writer.add_scalar('train/total_loss', float(np.mean(loss_list)), epoch)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./config/config.py')
    args = parser.parse_args()
    
    cfg = load_config(args.input)

    return cfg

def main():
    cfg = get_config()

    ###################
    # init parameters #
    ###################
    # creat save path 
    if os.path.exists(cfg.path.save):
        shutil.rmtree(cfg.path.save)
    os.makedirs(cfg.path.save, exist_ok=True)
    # log 
    logging.basicConfig(filename=cfg.path.save+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg))
    logging.info('--- init parameters ---')

    # training data path
    train_data_path = cfg.path.root_path

    # writer
    idx = cfg.path.save.rfind('/')
    log_dir = cfg.path.log_dir + cfg.path.save[idx:]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    
    # set title of the current process
    setproctitle.setproctitle(cfg.path.save)

    # random
    if cfg.training.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)
        torch.cuda.manual_seed(cfg.training.seed)


    #####################
    # building networks #
    #####################
    logging.info('--- building network ---')

    if cfg.general.arch == 'denseunet':
        net = ADenseUnet(in_channels=cfg.general.in_channels, num_classes=cfg.general.num_classes)
    elif cfg.general.arch == 'vnet':
        net = VNet(n_channels=cfg.general.in_channels, n_classes=cfg.general.num_classes)
    else:
        raise(RuntimeError('No module named {}'.fomat(cfg.general.arch)))

    if cfg.training.ngpu > 1:
        net = nn.parallel.DataParallel(net, list(range(cfg.training.ngpu)))

    n_params = sum([p.data.nelement() for p in net.parameters()])
    logging.info('total parameters = {}'.format(n_params))
    
    net = net.cuda()


    #####################
    # preparing dataset #
    #####################
    logging.info('--- loading dataset ---')

    train_transform = transforms.Compose([
        RandomRotFlip(),
        RandomCrop(cfg.general.crop_size),
        ToTensor()
        ])
    test_transform = transforms.Compose([ToTensor()])
    train_set = ABUS(base_dir=cfg.path.root_path,
                     split='train',
                     fold=cfg.training.fold,
                     transform=train_transform
                     )
    test_set = ABUS(base_dir=cfg.path.root_path,
                    split='test',
                    fold=cfg.testing.fold,
                    transform=test_transform
                    )
    kwargs = {'num_workers': cfg.training.num_workers, 'pin_memory': cfg.training.pin_memory}
    batch_size = cfg.training.batch_size * cfg.training.ngpu
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    batch_size = cfg.testing.batch_size
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)


    #####################
    # optimizer & loss  #
    #####################
    logging.info('--- configing optimizer & losses ---')
    
    lr = cfg.training.lr
    if cfg.training.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
    else:
        raise(RuntimeError('no optimizer named {}'.fomat(cfg.training.opt)))

    loss_fn = {}
    loss_fn['dice_loss'] = dice_loss 


    #####################
    #   strat training  #
    #####################
    logging.info('--- start training ---')

    best_pre = 0.
    for epoch in range(cfg.training.start_epoch, cfg.training.n_epochs + 1):
        # update lr
        if cfg.training.opt == 'adam':
            if epoch % 30 == 0:
                if epoch % 60 == 0:
                    lr *= 0.2
                else:
                    lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        writer.add_scalar('training/lr', lr, epoch)

        # train and evaluate 
        train(cfg, epoch, net, train_loader, optimizer, loss_fn, writer)
        #dice = test(cfg, epoch, net, test_loader, writer)
        dice = 0

        # save checkpoint
        if epoch % 10 == 0:
            is_best = False
            if dice > best_pre:
                is_best = True
                best_pre = dice
            save_checkpoint({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_pre':best_pre}, 
                            is_best, 
                            cfg.path.save, 
                            cfg.training.arch)

    writer.close()

def save_checkpoint(state, is_best, path, prefix, filename='checkpoing.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

if __name__ == "__main__":
    main()
