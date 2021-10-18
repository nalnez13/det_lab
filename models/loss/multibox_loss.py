import math
from models.loss.focal_loss import FocalLoss
from models.detector.anchors import Anchors
import torch
from torch import nn


def calc_iou(anchors, boxes):
    """calculate ious of anchor x boxes
        loc format : x1, y1, x2, y2
    Args:
        anchors (tensor): [Na, 4] (Na: Anchor 수)
        boxes (tensor): [Nb, 4] (Nb: 해당 Sample의 GT Box 수)
    Returns:
        tensor: IoU Values [Na, Nb]
    """
    anchors_area = torch.unsqueeze(
        (anchors[:, 2] - anchors[:, 0]) *
        (anchors[:, 3] - anchors[:, 1]),
        dim=1)
    boxes_area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    min_x2 = torch.min(torch.unsqueeze(anchors[:, 2], dim=1), boxes[:, 2])
    max_x1 = torch.max(torch.unsqueeze(anchors[:, 0], dim=1), boxes[:, 0])
    min_y2 = torch.min(torch.unsqueeze(anchors[:, 3], dim=1), boxes[:, 3])
    max_y1 = torch.max(torch.unsqueeze(anchors[:, 1], dim=1), boxes[:, 1])

    inter_w = min_x2 - max_x1
    inter_h = min_y2 - max_y1
    inter_w = torch.clamp(inter_w, min=0)
    inter_h = torch.clamp(inter_h, min=0)
    inter_area = inter_w * inter_h

    union = anchors_area + boxes_area - inter_area
    union = torch.clamp(union, min=1e-7)

    IoU = inter_area / union
    return IoU


def smooth_l1_loss(positive_anchors, gt_samples, prediction):
    anchors_cx_pi = positive_anchors[0]
    anchors_cy_pi = positive_anchors[1]
    anchors_width_pi = positive_anchors[2]
    anchors_height_pi = positive_anchors[3]

    gt_cx = gt_samples[0]
    gt_cy = gt_samples[1]
    gt_width = gt_samples[2]
    gt_height = gt_samples[3]

    dx = (gt_cx - anchors_cx_pi) / anchors_width_pi
    dy = (gt_cy - anchors_cy_pi) / anchors_height_pi
    dw = torch.log(gt_width / anchors_width_pi)
    dh = torch.log(gt_height / anchors_height_pi)

    targets = torch.stack((dx, dy, dw, dh))

    diff = torch.abs(targets - prediction)
    reg_loss = torch.where(
        torch.le(diff, 1/9.),
        0.5 * 9.0 * torch.pow(diff, 2),
        diff - 0.5 / 9.
    )
    return reg_loss


def cIoU_loss(positive_anchors, gt_samples, prediction):
    anchors_cx_pi = positive_anchors[0]
    anchors_cy_pi = positive_anchors[1]
    anchors_width_pi = positive_anchors[2]
    anchors_height_pi = positive_anchors[3]

    gt_cx = gt_samples[0]
    gt_cy = gt_samples[1]
    gt_width = gt_samples[2]
    gt_height = gt_samples[3]

    pred_cx = anchors_cx_pi + prediction[0] * anchors_width_pi
    pred_cy = anchors_cy_pi + prediction[1] * anchors_height_pi
    pred_width = torch.exp(prediction[2]) * anchors_width_pi
    pred_height = torch.exp(prediction[3]) * anchors_height_pi

    gt_area = gt_width * gt_height
    pred_area = pred_width * pred_height

    inter_x1 = torch.max(gt_cx - gt_width * 0.5, pred_cx - pred_width * 0.5)
    inter_x2 = torch.min(gt_cx + gt_width * 0.5, pred_cx + pred_width * 0.5)
    inter_y1 = torch.max(gt_cy - gt_height * 0.5, pred_cy - pred_height * 0.5)
    inter_y2 = torch.min(gt_cy + gt_height * 0.5, pred_cy + pred_height * 0.5)
    inter_area = torch.clamp((inter_x2 - inter_x1), min=0) * \
        torch.clamp((inter_y2 - inter_y1), min=0)

    enclosure_x1 = torch.min(gt_cx - gt_width * 0.5,
                             pred_cx - pred_width * 0.5)
    enclosure_x2 = torch.max(gt_cx + gt_width * 0.5,
                             pred_cx + pred_width * 0.5)
    enclosure_y1 = torch.min(gt_cy - gt_height * 0.5,
                             pred_cy - pred_height * 0.5)
    enclosure_y2 = torch.max(gt_cy + gt_height * 0.5,
                             pred_cy + pred_height * 0.5)

    inter_diag = (gt_cx - pred_cx)**2 + (gt_cy - pred_cy)**2
    enclosure_diag = torch.clamp(enclosure_x2 - enclosure_x1, min=0)**2 + \
        torch.clamp(enclosure_y2 - enclosure_y1, min=0)**2

    union = gt_area + pred_area - inter_area
    u = inter_diag / torch.clamp(enclosure_diag, min=1e-6)
    iou = inter_area / torch.clamp(union, min=1e-6)
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(gt_width / gt_height) -
                                         torch.atan(pred_width / pred_height), 2)
    print(iou)
    with torch.no_grad():
        S = (iou > 0.5).float()
        alpha = S * v / torch.clamp(1 - iou + v, min=1e-6)
    cIoU = iou - u - alpha * v
    cIoU = torch.clamp(cIoU, min=-1.0, max=1.0)
    return 1 - cIoU


class MultiBoxLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.anchors = Anchors()
        self.focal = FocalLoss()
        self.reg_loss = None

        if cfg['reg_loss'] == 'ciou_loss':
            self.reg_loss = cIoU_loss
        else:  # 'smooth_l1'
            self.reg_loss = smooth_l1_loss

        self.std = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).t()

    def forward(self, x):
        """calculate loss

        Args:
            x (list): [cls_pred, reg_pred, sample(dict)]

        """
        cls_pred, reg_pred, samples = x
        batch_size = cls_pred.shape[0]
        imgs = samples['img']
        annots = samples['annot']
        anchors = self.anchors(imgs)
        std = self.std.type_as(anchors)
        reg_pred = reg_pred * std
        device = imgs.device

        anchors_width = anchors[:, 2] - anchors[:, 0]
        anchors_height = anchors[:, 3] - anchors[:, 1]
        anchors_cx = anchors[:, 0] + 0.5*anchors_width
        anchors_cy = anchors[:, 1] + 0.5*anchors_height

        cls_pred = torch.clamp(cls_pred, 1e-7, 1. - 1e-7)

        cls_losses = []
        reg_losses = []

        for b in range(batch_size):
            classification = cls_pred[b]
            regression = reg_pred[b]
            bboxes = annots[b]
            bboxes = bboxes[bboxes[:, 4] != -1]

            targets = torch.zeros_like(classification)
            # no gt box
            if bboxes.shape[0] == 0:
                cls_loss = self.focal([classification, targets])
                cls_loss = cls_loss.sum()
                cls_losses.append(cls_loss)
                reg_losses.append(torch.tensor(0).to(device).float())
                continue

            # sampling positive samples
            IoU = calc_iou(anchors, bboxes[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            positive_samples = torch.ge(IoU_max, 0.5)
            assigned_bboxes = bboxes[IoU_argmax, :]

            # calculate classification loss
            targets[assigned_bboxes[positive_samples, 4].long(),
                    positive_samples] = 1

            cls_loss = self.focal([classification, targets])
            cls_loss = cls_loss.sum() / \
                torch.clamp(positive_samples.sum().float(), min=1)
            cls_losses.append(cls_loss)

            # calculate regression loss
            if positive_samples.sum() > 0:
                anchors_width_pi = anchors_width[positive_samples]
                anchors_height_pi = anchors_height[positive_samples]
                anchors_cx_pi = anchors_cx[positive_samples]
                anchors_cy_pi = anchors_cy[positive_samples]

                positive_anchors = torch.stack(
                    [anchors_cx_pi, anchors_cy_pi,
                     anchors_width_pi, anchors_height_pi])

                assigned_bboxes = assigned_bboxes[positive_samples, :]
                gt_width = assigned_bboxes[:, 2] - assigned_bboxes[:, 0]
                gt_height = assigned_bboxes[:, 3] - assigned_bboxes[:, 1]
                gt_width = torch.clamp(gt_width, min=1)
                gt_height = torch.clamp(gt_height, min=1)
                gt_cx = assigned_bboxes[:, 0] + 0.5 * gt_width
                gt_cy = assigned_bboxes[:, 1] + 0.5 * gt_height

                gt_samples = torch.stack(
                    [gt_cx, gt_cy, gt_width, gt_height])

                loss = self.reg_loss(positive_anchors, gt_samples,
                                     regression[:, positive_samples])

                reg_losses.append(loss.mean())
            else:
                reg_losses.append(torch.tensor(0).to(device).float())

        return torch.stack(cls_losses).mean(), torch.stack(reg_losses).mean()
