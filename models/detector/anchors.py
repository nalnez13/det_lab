import torch
from torch import nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, sizes=None, ratios=None,
                 scales=None, strides=None):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.strides = strides
        self.size_factors = None

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        if sizes is None:
            # NOTE: RetinaNet은 800~1333 크기의 이미지에 32 ~ 512 크기의 Anchor를 사용함
            # 사용할 모델은 작은 해상도를 사용하므로 대응되지 않음
            # 입력 이미지에 따라 대응되도록 SSD의 0.2~0.9 Size Factor 수식으로 대체함
            self.sizes = [2**(x+2) for x in self.pyramid_levels]
            min_size_scale = 0.2
            max_size_scale = 0.9
            self.size_factors = [
                min_size_scale + (max_size_scale - min_size_scale) /
                (len(self.pyramid_levels)-1) *
                x for x in range(len(self.pyramid_levels))]
            self.size_factors = torch.tensor(self.size_factors)

        if ratios is None:
            self.ratios = torch.tensor([0.5, 1, 2])

        if scales is None:
            self.scales = torch.tensor(
                [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        if strides is None:
            self.strides = [2**x for x in self.pyramid_levels]

        self.num_anchors = len(self.ratios) * len(self.scales)

    def forward(self, image):
        """Anchor Generation

        Args:
            image (Tensor): [C, H, W]의 Torch Tensor
        """
        img_shape = torch.tensor(image.shape[2:])
        img_shapes = [(img_shape + 2 ** x - 1) // (2**x)
                      for x in self.pyramid_levels]
        self.sizes = torch.mean(img_shape.float()) * self.size_factors

        all_anchors = torch.zeros((0, 4))
        for idx in range(len(self.pyramid_levels)):
            anchors = self.generate_anchors_per_pyramid(
                self.sizes[idx], self.scales, self.ratios)
            anchors = self.spread_anchors(
                img_shapes[idx], self.strides[idx], anchors)
            all_anchors = torch.cat((all_anchors, anchors), dim=0)
        return all_anchors.to(image.device)

    def generate_anchors_per_pyramid(self, base_size, scales, ratios):
        anchors = torch.zeros((self.num_anchors, 4))
        anchors[:, 2:] = base_size * torch.tile(scales, (2, len(ratios))).T

        areas = anchors[:, 2] * anchors[:, 3]

        anchors[:, 2] = torch.sqrt(areas/ratios.repeat(len(scales)))
        anchors[:, 3] = anchors[:, 2] * ratios.repeat(len(scales))

        anchors[:, 0::2] -= torch.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= torch.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    def spread_anchors(self, img_shape, stride, anchors):
        shift_x = (torch.arange(0, img_shape[1]) + 0.5) * stride
        shift_y = (torch.arange(0, img_shape[0]) + 0.5) * stride

        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)

        shifts = torch.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).T

        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = anchors.reshape(
            (1, A, 4)) + shifts.reshape(1, K, 4).permute(1, 0, 2)
        all_anchors = all_anchors.reshape((K*A, 4))
        return all_anchors


if __name__ == '__main__':
    a = Anchors()
    anchors = a(torch.zeros((1, 3, 320, 320)))
    print(anchors)
    print(anchors.shape)
