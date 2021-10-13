import torch
from models.detector.fpn import FeaturesPyramidNetwork
from torch import nn
import math

from models.initialize import weight_initialize


class ClassificationModel(nn.Module):
    def __init__(self, num_classes, in_features, num_anchors=9,
                 feature_size=256, prior=0.01):
        """Classfication Subnet

        Args:
            num_classes (int): 예측 Class 수
            in_features (int): input Feature Map Ch 수
            num_anchors (int, optional): 예측 Anchor 수. Defaults to 9.
            feature_size (int, optional): SubNet Convolution Filter 수. Defaults to 256.
            prior (float, optional): bias 초기화 Prior 값. Defaults to 0.01.
                                    https://arxiv.org/pdf/1708.02002.pdf
        """
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_features, feature_size,
                               kernel_size=3, stride=1, padding=1)
        # self.conv1_bn = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        # self.conv2_bn = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        # self.conv3_bn = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        # self.conv4_bn = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes,
                                kernel_size=3, stride=1, padding=1)
        self.output_act = nn.Sigmoid()

        weight_initialize(self)
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(-math.log((1.0 - prior)/prior))

    def forward(self, x):
        """classifcation forward

        Args:
            x (tensor): feature map of backbone

        Returns:
            [tensor]: prediction of classification
                      (b, num_classes, num_anchors * h * w)
        """
        out = self.conv1(x)
        # out = self.conv1_bn(out)
        out = self.act1(out)

        out = self.conv2(out)
        # out = self.conv2_bn(out)
        out = self.act2(out)

        out = self.conv3(out)
        # out = self.conv3_bn(out)
        out = self.act3(out)

        out = self.conv4(out)
        # out = self.conv4_bn(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        b, c, h, w = out.shape
        return out.contiguous().view(b, self.num_classes, -1)


class RegressionModel(nn.Module):
    def __init__(self, in_features, num_anchors=9, feature_size=256):
        """Regression SubNet

        Args:
            in_features (int): input Feature Map Ch 수
            num_anchors (int, optional): 예측 Anchor 수. Defaults to 9.
            feature_size (int, optional): SubNet Convolution Filter 수. Defaults to 256.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, feature_size,
                               kernel_size=3, stride=1, padding=1)
        # self.conv1_bn = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        # self.conv2_bn = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        # self.conv3_bn = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        # self.conv4_bn = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4,
                                kernel_size=3, stride=1, padding=1)

        weight_initialize(self)
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(0)

    def forward(self, x):
        """Regression Forward

        Args:
            x (tensor): Feature map of backbone

        Returns:
            tensor: Prediction of regression [b, 4, num_anchors*h*w]
        """
        out = self.conv1(x)
        # out = self.conv1_bn(out)
        out = self.act1(out)

        out = self.conv2(out)
        # out = self.conv2_bn(out)
        out = self.act2(out)

        out = self.conv3(out)
        # out = self.conv3_bn(out)
        out = self.act3(out)

        out = self.conv4(out)
        # out = self.conv4_bn(out)
        out = self.act4(out)

        out = self.output(out)
        b, c, h, w = out.shape
        return out.contiguous().view(b, 4, -1)


class RetinaNet(nn.Module):

    def __init__(self, Backbone, FPN, ClassificationSubNet, RegressionSubNet,
                 num_classes, in_channels=3):
        super().__init__()

        self.backbone = Backbone(in_channels)

        fpn_sizes = self.backbone.stage_channels[2:]
        self.fpn = FPN(fpn_sizes)

        feature_size = self.fpn.feature_size
        self.classification = ClassificationSubNet(
            num_classes, in_features=feature_size)

        self.regression = RegressionSubNet(in_features=feature_size)

    def forward(self, x):
        # backbone forward
        x = self.backbone.stem(x)
        s1 = self.backbone.layer1(x)
        s2 = self.backbone.layer2(s1)
        s3 = self.backbone.layer3(s2)
        s4 = self.backbone.layer4(s3)
        s5 = self.backbone.layer5(s4)

        # fpn forward
        features = self.fpn(s3, s4, s5)

        # prediction
        classifications = torch.cat(
            [self.classification(f) for f in features], dim=2)
        regressions = torch.cat([self.regression(f) for f in features], dim=2)

        return classifications, regressions


if __name__ == '__main__':
    from models.backbone.frostnet import FrostNet
    model = RetinaNet(
        Backbone=FrostNet,
        FPN=FeaturesPyramidNetwork,
        ClassificationSubNet=ClassificationModel,
        RegressionSubNet=RegressionModel,
        num_classes=20
    )
    print(model(torch.rand(1, 3, 320, 320)))
