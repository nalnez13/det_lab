from torch import nn
from torch.nn.modules.activation import Hardsigmoid


class SEBlockHardSigmoid(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(SEBlockHardSigmoid, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(
            in_features=in_channels, out_features=in_channels//4)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(
            in_features=in_channels//4, out_features=in_channels)
        self.act2 = Hardsigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x)
        y = y.view(b, c)
        y = self.linear1(y)
        y = self.act1(y)
        y = self.linear2(y)
        y = self.act2(y)
        y = y.view(b, c, 1, 1)
        return x * y
