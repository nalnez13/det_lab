from torch import nn


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * self.relu6(x+3) / 6
