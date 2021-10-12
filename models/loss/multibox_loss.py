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


class MultiBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchors = Anchors()
        self.focal = FocalLoss()

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

                assigned_bboxes = assigned_bboxes[positive_samples, :]
                gt_width = assigned_bboxes[:, 2] - assigned_bboxes[:, 0]
                gt_height = assigned_bboxes[:, 3] - assigned_bboxes[:, 1]
                gt_cx = assigned_bboxes[:, 0] + 0.5 * gt_width
                gt_cy = assigned_bboxes[:, 1] + 0.5 * gt_height
                gt_width = torch.clamp(gt_width, min=1)
                gt_height = torch.clamp(gt_height, min=1)

                dx = (gt_cx - anchors_cx_pi) / anchors_width_pi
                dy = (gt_cy - anchors_cy_pi) / anchors_height_pi
                dw = torch.log(gt_width / anchors_width_pi)
                dh = torch.log(gt_height / anchors_height_pi)

                targets = torch.stack((dx, dy, dw, dh))
                targets = targets / \
                    torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device).t()

                diff = torch.abs(targets - regression[:, positive_samples])
                reg_loss = torch.where(
                    torch.le(diff, 1/9.),
                    0.5 * 9.0 * torch.pow(diff, 2),
                    diff - 0.5 / 9.
                )
                reg_losses.append(reg_loss.mean())
            else:
                reg_losses.append(torch.tensor(0).to(device).float())

        return torch.stack(cls_losses).mean() + torch.stack(reg_losses).mean()
