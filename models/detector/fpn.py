from torch import nn


class FeaturesPyramidNetwork(nn.Module):
    def __init__(self, fpn_sizes, feature_size=256):
        """FPN 구현

        Args:
            fpn_sizes (list): [s3, s4, s5] stage의 output ch 값
            feature_size (int, optional): FPN 내부 Convoluiton Filter 수
                                          Defaults to 256.
        """
        super().__init__()
        self.feature_size = feature_size
        s3_size, s4_size, s5_size = fpn_sizes
        self.P5_1 = nn.Conv2d(s5_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P5_1_bn = nn.BatchNorm2d(feature_size)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        self.P5_2_bn = nn.BatchNorm2d(feature_size)

        self.P4_1 = nn.Conv2d(s4_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P4_1_bn = nn.BatchNorm2d(feature_size)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        self.P4_2_bn = nn.BatchNorm2d(feature_size)

        self.P3_1 = nn.Conv2d(s3_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P3_1_bn = nn.BatchNorm2d(feature_size)
        self.P3_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        self.P3_2_bn = nn.BatchNorm2d(feature_size)

        self.P6 = nn.Conv2d(feature_size, feature_size,
                            kernel_size=3, stride=2, padding=1)
        self.P6_bn = nn.BatchNorm2d(feature_size)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=2, padding=1)
        self.P7_2_bn = nn.BatchNorm2d(feature_size)

    def forward(self, s3, s4, s5):
        """FPN Forward

        Args:
            s3, s4, s5 ([tensor]): feature map stages

        Returns:
            [list]: fused feature maps of stages [p3, p4, p5, p6, p7]
        """

        P5_x = self.P5_1(s5)
        P5_x = self.P5_1_bn(P5_x)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        P5_x = self.P5_2_bn(P5_x)

        P4_x = self.P4_1(s4)
        P4_x = self.P4_1_bn(P4_x)
        P4_x = P4_x + P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        P4_x = self.P4_2_bn(P4_x)

        P3_x = self.P3_1(s3)
        P3_x = self.P3_1_bn(P3_x)
        P3_x = P3_x+P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        P3_x = self.P3_2_bn(P3_x)

        P6_x = self.P6(P5_x)
        P6_x = self.P6_bn(P6_x)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        P7_x = self.P7_2_bn(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
