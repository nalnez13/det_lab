from .layers.conv_block import Conv2dBn, Conv2dBnRelu
from .initialize import weight_initialize
from utils.utility import make_divisible
import torch
from torch import nn


class FrostConv(nn.Module):
    def __init__(self, in_channels, reduction_factor, expansion_factor,
                 out_channels, kernel_size, stride):
        super(FrostConv, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        reduction = make_divisible(in_channels//reduction_factor)
        expansion = (reduction + in_channels) * expansion_factor
        self.squeeze = Conv2dBnRelu(in_channels, reduction, 1)
        self.expand = Conv2dBnRelu(in_channels+reduction, expansion, 1)
        self.depthwise = Conv2dBnRelu(
            expansion, expansion, kernel_size, stride, groups=expansion)
        self.projection = Conv2dBn(expansion, out_channels, 1)

    def forward(self, x):
        squeeze = self.squeeze(x)
        y = torch.cat((x, squeeze), dim=1)
        y = self.expand(y)
        y = self.depthwise(y)
        y = self.projection(y)
        if self.use_residual:
            y = x + y
        return y


class _FrostNetSmall(nn.Module):
    def __init__(self, in_channels, classes, width_mult=1.0):
        super(_FrostNetSmall, self).__init__()
        stem_channels = make_divisible(int(32 * min(1.0, width_mult)))
        self.in_channels = stem_channels
        self.width_mult = width_mult
        self.stem = Conv2dBnRelu(in_channels, stem_channels, 3, 2)

        # config kernel_size, output_ch, ef, rfs, stride
        layer1 = [
            [3, 16, 1, 1, 1]]
        layer2 = [
            [5, 24, 6, 4, 2],
            [3, 24, 3, 4, 1]
        ]
        layer3 = [
            [5, 40, 3, 4, 2],
            [5, 40, 3, 4, 1]
        ]
        layer4 = [
            [5, 80, 3, 4, 2],
            [3, 80, 3, 4, 1],
            [5, 96, 3, 2, 1],
            [3, 96, 3, 4, 1],
            [5, 96, 3, 4, 1],

        ]
        layer5 = [
            [5, 192, 6, 2, 2],
            [5, 192, 3, 2, 1],
            [5, 192, 3, 2, 1],
            [5, 320, 6, 2, 1],
        ]
        self.layer1 = self.make_layers(layer1)
        self.layer2 = self.make_layers(layer2)
        self.layer3 = self.make_layers(layer3)
        self.layer4 = self.make_layers(layer4)
        self.layer5 = self.make_layers(layer5)
        self.classification = nn.Sequential(
            Conv2dBnRelu(self.in_channels, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, x):
        y = self.stem(x)
        s1 = self.layer1(y)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        stages = [s1, s2, s3, s4, s5]

        return {'stages': stages, 'pred': pred}

    def make_layers(self, layer_configs):
        layers = []
        for k, o, ef, rf, s in layer_configs:
            out_channels = make_divisible(o * self.width_mult)
            layers.append(FrostConv(self.in_channels,
                          rf, ef, out_channels, k, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


def FrostNet(in_channels, classes, varient='small'):
    if varient == 'small':
        model = _FrostNetSmall(in_channels, classes)

    else:
        raise Exception('No such models {}'.format(varient))

    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = FrostNet(in_channels=3, classes=1000)
    print(model(torch.rand(1, 3, 224, 224)))
