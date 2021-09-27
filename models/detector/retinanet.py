from torch import nn
import math


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
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes,
                                kernel_size=3, stride=1, padding=1)
        self.output_act = nn.Sigmoid()

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
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        b, c, h, w = out.shape
        return out.contigous().view(b, self.num_classes, -1)


class RegressionModel(nn.Module):
    def __init__(self, in_features, num_anchors=9, feature_size=256):
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, feature_size,
                               kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4,
                                kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Regression Forward

        Args:
            x (tensor): Feature map of backbone

        Returns:
            tensor: Prediction of regression [b, 4, num_anchors*h*w]
        """
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        b, c, h, w = out.shape
        return out.contigous().view(b, 4, -1)
