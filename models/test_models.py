import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
from atrous_denseunet import AtrousDenseNet, ADenseUnet
from vnet_o import VNet 
import torch
from tensorboardX import SummaryWriter
#from unet3d.model import UNet3D
from unet import UNet3D

#net = VNet(n_channels=1, n_classes=2)
net = ADenseUnet()
#net = UNet3D(residual=False)
#net = UNet3D(residual=True)
#net = net.cuda()
print(net)
n_params = sum([p.data.nelement() for p in net.parameters()])
print('total parameters = {}'.format(n_params))
#
#x = torch.rand((1, 1, 64, 128, 128))
#x = x.cuda()
#writer = SummaryWriter('../log/0305_archs/denseunet-arch')
#writer.add_graph(net, (x, ))
#writer.close()
