from torch import nn
import numpy as np


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, sizes=None, ratios=None,
                 scales=None, strides=None):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.strides = strides

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        if sizes is None:
            # TODO: Image 크기랑 관계없이 32 ~ 512 크기의 Anchor를 사용하는데...
            # 이미지 크기에 따라 가변돼야 할 듯.
            # RetinaNet은 800~1333 정도 크기 이미지를 입력받음
            # 적절한 Size Mapping 수식 필요
            self.sizes = [2**(x+2) for x in self.pyramid_levels]

        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])

        if scales is None:
            self.scales = np.array(
                [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        if strides is None:
            self.strides = [2**x for x in self.pyramid_levels]

        self.num_anchors = len(self.ratios) * len(self.scales)

    def forward(self, image):
        """Anchor Generation

        Args:
            image (Tensor): [C, H, W]의 Torch Tensor
        """
        img_shape = np.array(image.shape[1:])
        img_shapes = [(img_shape + 2 ** x - 1) // (2**x)
                      for x in self.pyramid_levels]

        for idx in range(len(self.pyramid_levels)):
            anchors = self.generate_anchors_per_pyramid(
                self.sizes[idx], self.scales, self.ratios)
            self.spread_anchors(img_shapes[idx], self.strides[idx], anchors)

    def generate_anchors_per_pyramid(self, base_size, scales, ratios):
        anchors = np.zeros((self.num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        areas = anchors[:, 2] * anchors[:, 3]
        anchors[:, 2] = np.sqrt(areas/np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    def spread_anchors(self, img_shape, stride, anchors):
        shift_x = (np.arange(0, img_shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, img_shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).T

        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = anchors.reshape(
            (1, A, 4)) + shifts.reshape(1, K, 4).transpose(1, 0, 2)
        all_anchors = all_anchors.reshape((K*A, 4))
        # print(all_anchors)


if __name__ == '__main__':
    a = Anchors()
    print(a(np.zeros((3, 320, 320))))
