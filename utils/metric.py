from medpy import metric

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

