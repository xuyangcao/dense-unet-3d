import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from torch.autograd import Variable
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance

class CertaintyLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(CertaintyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, uncertainty):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        #print('target.shape: ', target.shape)
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        #pt = Variable(logpt.data.exp())
        uncertainty = uncertainty.view(-1)
        uncertainty = uncertainty.cuda()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (uncertainty)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class MaskMSELoss(nn.Module):
    def __init__(self):
        super(MaskMSELoss, self).__init__()

    def forward(self, out, zcomp, uncer, th=0.15):
        # transverse to float 
        out = out.float()
        zcomp = zcomp.float()
        uncer = uncer.float()
        mask = uncer > th
        mask = mask.float()
        mse = torch.sum(mask*(out - zcomp)**2) / torch.sum(mask) 

        return mse

def generalized_distance_transform(inputs):
    in_shape = inputs.shape 
    
    i = torch.arange(in_shape[1]).float()
    j = torch.arange(in_shape[2]).float()

    coords = torch.stack(torch.meshgrid(i, j), -1)
    print('coords: ', coords.shape)

    d_pq = torch.norm(torch.reshape(coords, (-1, 1, 2))
                     -torch.reshape(coords, (1, -1, 2)), dim=-1, keepdim=True)

    d_pq = d_pq.cuda()
    f_q = torch.reshape(inputs, (-1, 1, in_shape[1]*in_shape[2], in_shape[3]))
    print('d_pq.shape: ', d_pq.shape)
    print('fq.shape: ', f_q.shape)

    dt = torch.mean(d_pq + f_q, dim=2)
    dt = torch.reshape(dt, in_shape)
    return dt

class GeneralizedDiceLoss(nn.Module):
    # generalized Dice Loss
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, pre, gt):
        num_classes = pre.shape[1]
        # get one hot ground truth
        gt = gt.long()
        gt_one_hot = torch.eye(num_classes)[gt.squeeze(1)] # [1, 128, 128, 128, 2]
        #print('gt_one_hot.shape: ', gt_one_hot.shape)
        #print('pre.shapre: ', pre.shape)
        gt_one_hot = gt_one_hot.permute(0, 4, 1, 2, 3).float() # [1, 2, 128, 128, 128]
        if gt.cuda:
            gt_one_hot = gt_one_hot.cuda()
        pre = pre.float()

        # dice loss
        eps = 1e-6
        dims = (0,) + tuple(range(2, gt.ndimension())) # (0, 2, 3, 4)
        intersection = torch.sum(pre * gt_one_hot, dims)
        cardinality = torch.sum(pre + gt_one_hot, dims)
        dice = (2. * intersection / (cardinality + eps)).mean()
        loss = 1. - dice

        return loss

    @staticmethod
    def dice_coeficient(output, target):
        output = output.float()
        target = target.float()
        
        output = output
        smooth = 1e-20
        iflat = output.view(-1)
        tflat = target.view(-1)
        
        intersection = torch.dot(iflat, tflat)
        dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

        return dice 

def threshold_loss(pred, threshold_map, target):
    scalar = -torch.exp(torch.tensor([10.])) 
    scalar = scalar.cuda()
    temp = pred.clone()
    temp[pred < threshold_map] = scalar
    #mask = pred > threshold_map
    #mask = mask.float()
    mask = torch.sigmoid(temp)
    loss = dice_loss(mask, target)

    return loss

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def focal_dice_loss(score, target, gamma=2.):
    target = target.float()
    smooth = 1e-6
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    loss = - (1 - dice) ** gamma * torch.log(dice + smooth) 

    return loss

def dice_loss1(score, target):
    # non-square
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:,1,...]
    dc = gt_sdf[:,0,...]
    multipled = torch.einsum('bxyz, bxyz->bxyz', pc, dc)
    bd_loss = multipled.mean()

    return bd_loss

def iou_loss(score, target):
    target = target.float()
    smooth = 1e-5
    tp_sum = torch.sum(score * target)
    fp_sum = torch.sum(score * (1-target))
    fn_sum = torch.sum((1-score) * target)
    loss = (tp_sum + smooth) / (tp_sum + fp_sum + fn_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis - posdis
            sdf[boundary==1] = 0
            gt_sdf[b][c] = sdf

    return gt_sdf


def compute_bounds(target, out_shape, foreground=-5, background=1):
    gt_sdf = np.zeros(out_shape)
    if target.sum() == 0:
        return gt_sdf 
    else:
        bounds = target.copy()
        bounds = bounds.astype(np.float32)
        bounds[bounds != 0] = foreground 
        bounds[bounds == 0] = background 

        gt_sdf[0] = bounds
        gt_sdf[1] = bounds

        return gt_sdf 

def compute_sdf01(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation

    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4: # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]): # batch size
        for c in range(dis_id, segmentation.shape[1]): # class_num
            # ignore background
            posmask = segmentation[b][c]
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis/np.max(negdis)/2 - posdis/np.max(posdis)/2 + 0.5
            sdf[boundary>0] = 0.5
            normalized_sdf[b][c] = sdf
    return normalized_sdf

def compute_sdf1_1(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation

    """
    # print(type(segmentation), segmentation.shape)

    img_gt = img_gt.astype(np.uint8)
    if img_gt.shape != out_shape:
        img_gt = np.expand_dims(img_gt, 1)
    #print('img_gt.shape: ', img_gt.shape)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
            # ignore background
        for c in range(out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b][c] = sdf
            assert np.min(sdf) == -1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
            assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def compute_sdf1_1_ori(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation

    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4: # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]): # batch size
        for c in range(dis_id, segmentation.shape[1]): # class_num
            # ignore background
            posmask = segmentation[b][c]
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis/np.max(negdis) - posdis/np.max(posdis)
            sdf[boundary>0] = 0
            normalized_sdf[b][c] = sdf
    return normalized_sdf


def compute_fore_dist(segmentation):
    """
    compute the foreground of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4: # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]): # batch size
        for c in range(dis_id, segmentation.shape[1]): # class_num
            # ignore background
            posmask = segmentation[b][c]
            posdis = distance(posmask)
            normalized_sdf[b][c] = posdis/np.max(posdis)
    return normalized_sdf


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def AAAI_sdf_loss(net_output, gt):
    """
    net_output: net logits; shape=(batch_size, class, x, y, z)
    gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
    """
    smooth = 1e-5
    axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
        gt_sdm_npy = compute_sdf1_1(y_onehot.cpu().numpy())
        gt_sdm = torch.from_numpy(gt_sdm_npy).float().cuda(net_output.device.index)
    intersect = sum_tensor(net_output * gt_sdm, axes, keepdim=False)
    pd_sum = sum_tensor(net_output ** 2, axes, keepdim=False)
    gt_sum = sum_tensor(gt_sdm ** 2, axes, keepdim=False)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum)
    # print('L_product.shape', L_product.shape) (4,2)
    L_SDF_AAAI = - L_product.mean() + torch.norm(net_output - gt_sdm, 1)/torch.numel(net_output)

    return L_SDF_AAAI


def sdf_kl_loss(net_output, gt):
    """
    net_output: net logits; shape=(batch_size, class, x, y, z)
    gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
    """
    smooth = 1e-5
    axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
        # print('y_onehot.shape', y_onehot.shape)
        gt_sdf_npy = compute_sdf(y_onehot.cpu().numpy())
        gt_sdf = torch.from_numpy(gt_sdf_npy+smooth).float().cuda(net_output.device.index)
    # print('net_output, gt_sdf', net_output.shape, gt_sdf.shape)
    # exit()
    sdf_kl_loss = F.kl_div(net_output, gt_sdf[:,1:2,...], reduction='batchmean')

    return sdf_kl_loss

