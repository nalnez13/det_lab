from torch import nn
import torch


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x):
        """calculate focal loss

        Args:
            x ([tensor]): prediction, target logits
        """
        pred, target = x
        alpha_factor = torch.where(
            torch.eq(target, 1.), self.alpha, 1.-self.alpha)
        focal_weight = torch.where(torch.eq(target, 1.), 1.-pred, pred)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        binary_ce = -(target * torch.log(pred) +
                      (1.-target)*torch.log(1.-pred))
        loss = focal_weight*binary_ce
        return loss
