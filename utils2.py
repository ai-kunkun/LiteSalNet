import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F

# from torch.autograd import Variable
# import numpy as np


fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
fx = np.reshape(fx, (1, 1, 3, 3))
fy = np.reshape(fy, (1, 1, 3, 3))
fx = Variable(torch.from_numpy(fx)).cuda()
fy = Variable(torch.from_numpy(fy)).cuda()
contour_th = 1.5


def label_edge_prediction(label):
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad


def pred_edge_prediction(pred):
    # infer edge from prediction
    pred = F.pad(pred, (1, 1, 1, 1), mode='replicate')
    pred_fx = F.conv2d(pred, fx)
    pred_fy = F.conv2d(pred, fy)
    pred_grad = (pred_fx * pred_fx + pred_fy * pred_fy).sqrt().tanh()

    return pred_grad


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def BER(pre_path, label_path):
    img_list = os.listdir(pre_path)
    sum_tp = 0.0
    sum_tn = 0.0
    sum_fp = 0.0
    sum_fn = 0.0
    for i, name in enumerate(img_list):
        if name.endswith('.png'):
            predict = cv2.imread(os.path.join(pre_path, name), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
            y_actual = torch.from_numpy(label).float()
            y_hat = torch.from_numpy(predict).float()
            y_hat = y_hat.ge(128).float()
            y_actual = y_actual.ge(128).float()
            y_actual = y_actual.squeeze(1)
            y_hat = y_hat.squeeze(1)
            pred_p = y_hat.eq(1).float()
            pred_n = y_hat.eq(0).float()
            pre_positive = float(pred_p.sum())
            pre_negtive = float(pred_n.sum())
            # FN
            fn_mat = torch.gt(y_actual, pred_p)
            FN = float(fn_mat.sum())
            # FP
            fp_mat = torch.gt(pred_p, y_actual)
            FP = float(fp_mat.sum())
            TP = pre_positive - FP
            TN = pre_negtive - FN
            sum_tp = sum_tp + TP
            sum_tn = sum_tn + TN
            sum_fp = sum_fp + FP
            sum_fn = sum_fn + FN
    pos = sum_tp + sum_fn
    neg = sum_tn + sum_fp

    if (pos != 0 and neg != 0):
        BAC = (.5 * ((sum_tp / pos) + (sum_tn / neg)))
    elif (neg == 0):
        BAC = (.5 * (sum_tp / pos))
    elif (pos == 0):
        BAC = (.5 * (sum_tn / neg))
    else:
        BAC = .5
    accuracy = (sum_tp + sum_tn) / (pos + neg) * 100
    global_ber = (1 - BAC) * 100
    return global_ber, accuracy


def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.cuda()
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge


def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx


def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy


def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001 ** 2, 0.5)
    return cp_s


def get_saliency_smoothness(pred, gt, size_average=True):
    alpha = 10
    s1 = 10
    s2 = 1
    ## first oder derivative: sobel
    sal_x = torch.abs(gradient_x(pred))
    sal_y = torch.abs(gradient_y(pred))
    gt_x = gradient_x(gt)
    gt_y = gradient_y(gt)
    w_x = torch.exp(torch.abs(gt_x) * (-alpha))
    w_y = torch.exp(torch.abs(gt_y) * (-alpha))
    cps_x = charbonnier_penalty(sal_x * w_x)
    cps_y = charbonnier_penalty(sal_y * w_y)
    cps_xy = cps_x + cps_y

    ## second order derivative: laplacian
    lap_sal = torch.abs(laplacian_edge(pred))
    lap_gt = torch.abs(laplacian_edge(gt))
    weight_lap = torch.exp(lap_gt * (-alpha))
    weighted_lap = charbonnier_penalty(lap_sal * weight_lap)

    smooth_loss = s1 * torch.mean(cps_xy) + s2 * torch.mean(weighted_lap)

    return smooth_loss


class smoothness_loss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(smoothness_loss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return get_saliency_smoothness(pred, target, self.size_average)

