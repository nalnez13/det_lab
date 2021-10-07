from torch import nn
import torch
from torchvision.ops import nms
from models.detector.anchors import Anchors


class Transformer(nn.Module):
    """Transform detector's logtis to bbox information
    """

    def __init__(self):
        super().__init__()
        self.anchors = Anchors()
        self.std = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).t()
        if torch.cuda.is_available:
            self.std = self.std.cuda()

    def forward(self, x):
        """
        Transformer Forward

        Args:
            x ([tensor]): [image, cls_pred, reg_pred]
        """
        image, cls_pred, reg_pred = x
        anchors = self.anchors(image)
        boxes_pred = self.regress_anchors(anchors, reg_pred)
        boxes_pred = self.clip_anchors(boxes_pred, image)
        boxes_pred = boxes_pred.permute(0, 2, 1)

        # for each classes
        num_samples = image.shape[0]
        scores_ret = [torch.empty(0, device=image.device)
                      for b in range(num_samples)]
        cls_ret = [torch.empty(0, device=image.device)
                   for b in range(num_samples)]
        boxes_ret = [torch.empty(0, device=image.device)
                     for b in range(num_samples)]
        for b in range(num_samples):
            for i in range(cls_pred.shape[1]):
                scores = torch.squeeze(cls_pred[b, i, :])
                over_thresh = scores > 0.05
                if over_thresh.sum() == 0:
                    continue
                scores = scores[over_thresh]
                boxes = boxes_pred[b, over_thresh, :]
                nms_idx = nms(boxes, scores, 0.5)
                scores_ret[b] = torch.cat((scores_ret[b], scores[nms_idx]))
                cls_values = torch.tensor([i] * nms_idx.shape[0])
                if torch.cuda.is_available:
                    cls_values = cls_values.cuda()
                cls_ret[b] = torch.cat((cls_ret[b], cls_values))
                boxes_ret[b] = torch.cat((boxes_ret[b], boxes[nms_idx]))
        return [scores_ret, cls_ret, boxes_ret]

    def regress_anchors(self, anchors, reg_pred):

        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        cx = anchors[:, 0] + 0.5 * widths
        cy = anchors[:, 1] + 0.5 * heights

        reg_pred = reg_pred * self.std

        dx = reg_pred[:, 0, :]
        dy = reg_pred[:, 1, :]
        dw = reg_pred[:, 2, :]
        dh = reg_pred[:, 3, :]

        pred_cx = cx + dx * widths
        pred_cy = cy + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_x1 = pred_cx - 0.5 * pred_w
        pred_x2 = pred_cx + 0.5 * pred_w
        pred_y1 = pred_cy - 0.5 * pred_h
        pred_y2 = pred_cy + 0.5 * pred_h

        pred_x1 = torch.clamp(pred_x1, min=0)

        return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

    def clip_anchors(self, anchors, img):
        b, c, h, w = img.shape
        anchors[:, 0, :] = torch.clamp(anchors[:, 0, :], min=0)
        anchors[:, 1, :] = torch.clamp(anchors[:, 1, :], min=0)
        anchors[:, 2, :] = torch.clamp(anchors[:, 2, :], max=w)
        anchors[:, 3, :] = torch.clamp(anchors[:, 3, :], max=h)
        return anchors
