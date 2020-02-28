import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
import sys
import tqdm
import random
import shutil
import logging
import argparse
import setproctitle
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
plt.switch_backend('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
from utils.metric import get_dice


def test(cfg, epoch, net, test_loader, writer):
    net.eval()
    mean_dice = []
    logging.info('--- testing ---')
    with torch.no_grad():
        for i, sample in tqdm.tqdm(enumerate(test_loader)):
            data, label = sample['image'], sample['label']
            data, label = data.cuda(), label.cuda()

            out = net(data)
            out = F.softmax(out, dim=1)
            out_new = out.max(1)[1]
            dice = get_dice(out_new.cpu().numpy(), label.cpu().numpy())
            mean_dice.append(dice)

            # show results on tensorboard
            if i % 10 == 0:
                nrow = 5
                image = data[0, 0:1, :, 30:71:10, :].permute(2,0,1,3)
                image = (image + 0.5) * 0.5
                grid_image = make_grid(image, nrow=nrow)
                grid_image = grid_image.cpu().detach().numpy().transpose((1,2,0))

                gt = label[0, :, 30:71:10, :].unsqueeze(0).permute(2,0,1,3)
                grid_gt = make_grid(gt, nrow=nrow, normalize=False)
                grid_gt = grid_gt.cpu().detach().numpy().transpose((1,2,0))[:, :, 0]
                gt_img = label2rgb(grid_gt, grid_image, bg_label=0)

                #outputs_soft = F.softmax(outputs, 1) #batchsize x num_classes x w x h x d
                pre = torch.max(out, dim=1, keepdim=True)[1]
                pre = pre[0, 0:1, :, 30:71:10, :].permute(2,0,1,3)
                grid_pre = make_grid(pre, nrow=nrow, normalize=False)
                grid_pre = grid_pre.cpu().detach().numpy().transpose((1,2,0))[:, :, 0]
                pre_img = label2rgb(grid_pre, grid_image, bg_label=0)

                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(gt_img, 'gray')
                ax.set_title('ground truth')
                ax = fig.add_subplot(212)
                ax.imshow(pre_img, 'gray')
                ax.set_title('prediction')
                fig.tight_layout()
                writer.add_figure('test/prediction_results', fig, epoch)
                fig.clear()
        
        writer.add_scalar('test/dice', float(np.mean(mean_dice)), epoch)
        return np.mean(mean_dice)


def train(cfg, epoch, net, train_loader, optimizer, loss_fn, writer):
    r"""
    training

    Args:
        cfg: config file
        epoch: current epoch
        net: network
        train_loader: train loader 
        optimizer: optimizer
        loss_fn: loss function
        writer: SummaryWriter

    Returns:
        None
    """
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
                grid_gt = grid_gt.cpu().detach().numpy().transpose((1,2,0))[:, :, 0]
                gt_img = label2rgb(grid_gt, grid_image, bg_label=0)

                #outputs_soft = F.softmax(outputs, 1) #batchsize x num_classes x w x h x d
                prob = outputs_soft[0, 1:2, :, 30:71:10, :].permute(2,0,1,3)
                grid_prob = make_grid(prob, nrow=nrow, normalize=False)
                grid_prob = grid_prob.cpu().detach().numpy().transpose((1,2,0))

                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(gt_img, 'gray')
                ax.set_title('ground truth')
               
                ax = fig.add_subplot(212)
                ax.set_title('prediction')
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                fig.add_axes(ax_cb)
                im = ax.imshow(grid_prob[:, :, 0], 'jet')
                plt.colorbar(im, cax=ax_cb)
                ax_cb.yaxis.tick_right()
                ax_cb.yaxis.set_tick_params(labelright=False)
                fig.tight_layout()
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
        raise(RuntimeError('No module named {}'.format(cfg.general.arch)))

    if cfg.training.ngpu > 1:
        net = nn.parallel.DataParallel(net, list(range(cfg.training.ngpu)))

    n_params = sum([p.data.nelement() for p in net.parameters()])
    logging.info('total parameters = {}'.format(n_params))
    
    net = net.cuda()
    # show graph
    #x = torch.rand((1, 1, 64, 128, 128)).cuda()
    #writer.add_graph(net, (x, ))


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
        writer.add_scalar('train/lr', lr, epoch)

        # train and evaluate 
        train(cfg, epoch, net, train_loader, optimizer, loss_fn, writer)
        if epoch == 1 or epoch % 20 == 0:
            dice = test(cfg, epoch, net, test_loader, writer)

            # save checkpoint
            is_best = False
            if dice > best_pre:
                is_best = True
                best_pre = dice
            save_checkpoint({'epoch': epoch,
                            'state_dict': net.state_dict(),
                            'best_pre':best_pre}, 
                            is_best, 
                            cfg.path.save, 
                            cfg.general.arch)

    writer.close()

def save_checkpoint(state, is_best, path, prefix, filename='checkpoing.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

if __name__ == "__main__":
    main()
