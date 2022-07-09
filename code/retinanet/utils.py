import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import scipy



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes


def loss_plot(losses,save_path):
    iters = range(len(losses))

    plt.figure()
    plt.plot(iters, losses, 'red', linewidth=2, label='train loss')

    try:
        if len(losses) < 25:
            num = 5
        else:
            num = 15

        plt.plot(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle='--', linewidth=2,
                 label='smooth train loss')

    except:
        pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(save_path, "epoch_loss"  + ".png"))
    plt.close()

def lr_plot(lrs,save_path):
    iters = range(len(lrs))
    plt.figure()
    plt.plot(iters, lrs, 'red', linewidth=2, label='learn rate')
    try:
        if len(lrs) < 25:
            num = 5
        else:
            num = 15

        plt.plot(iters, scipy.signal.savgol_filter(lrs, num, 3), 'green', linestyle='--', linewidth=2,
                 label='smooth train loss')

    except:
        pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('learn rate ')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(save_path, "learn_rate"  + ".png"))
    plt.close()


def mAP_plot(mAPs,class_dict,save_path):
    color=["blue", "green", "red", "cyan",  "magenta", "yellow" ,"black"]
    for i in range(len(class_dict)):
        color.append(np.random.randint(0,255))
    iters = range(len(mAPs['mAP']))
    plt.figure()
    index=0
    for key in mAPs.keys():
        value=mAPs[key]
        if key == 'mAP':
            plt.plot(iters, value, color[index], linewidth=2, label=key)
        else:
            plt.plot(iters, value, color[index], linewidth=2, label=class_dict[key] + ' AP')
        index+=1

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('AP_mAP')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(save_path, "valid mAP"  + ".png"))
    plt.close()


import numpy as np
def crop_first_nms(bboxes,classification,threshold=0.5, mode='union'):
    pass
    inds = np.arange(classification.shape[0])
    weed_inds = np.array([i for i in inds if classification.cpu()[i] ==0],dtype=int)
    crop_inds = np.array([i for i in inds if classification.cpu()[i] ==1],dtype=int)
    crop_x1 = bboxes[crop_inds, 0]
    crop_y1 = bboxes[crop_inds, 1]
    crop_x2 = bboxes[crop_inds, 2]
    crop_y2 = bboxes[crop_inds, 3]

    weed_x1 = bboxes[weed_inds, 0]
    weed_y1 = bboxes[weed_inds, 1]
    weed_x2 = bboxes[weed_inds, 2]
    weed_y2 = bboxes[weed_inds, 3]

    crop_areas = (crop_x2 - crop_x1 + 1) * (crop_y2 - crop_y1 + 1)
    weed_areas = (weed_x2-weed_x1+1) * (weed_y2-weed_y1+1)
    keep = np.array([],dtype=int)

    for i in range(len(crop_inds)):
        keep = np.append(keep,crop_inds[i])
        if len(weed_inds)==0:
            continue
        xx1 = weed_x1[weed_inds].clamp(min= crop_x1[i])
        yy1 = weed_y1[weed_inds].clamp(min=crop_y1[i])

        xx2 = weed_x2[weed_inds].clamp(max=crop_x2[i])
        yy2 = weed_y2[weed_inds].clamp(max=crop_y2[i])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        ovr = inter / (crop_areas[i] + weed_areas[weed_inds] - inter)
        ovr_idxs = np.where(ovr.cpu()<=threshold)[0] #(ovr>threshold).nonzero().squeeze().cpu()
        weed_inds = weed_inds[ovr_idxs]

    new_inds = (np.append(weed_inds,keep))

    return new_inds
