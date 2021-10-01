import argparse
import cv2
from numpy.lib.type_check import imag
import torch
from module.detector import Detector
from models.detector.retinanet import RetinaNet
from module.transformer import Transformer
from utils.module_select import get_cls_subnet, get_fpn, get_model, get_reg_subnet
import numpy as np
from utils.utility import preprocess_input
from utils.yaml_helper import get_train_configs


def main(cfg, image_name):
    # Preprocess Image
    image = cv2.imread(image_name)
    image = cv2.resize(image, (320, 320))
    image_inp = preprocess_input(image)
    image_inp = image_inp.unsqueeze(0)
    if torch.cuda.is_available:
        image_inp = image_inp.cuda()

    # Load trained model
    backbone = get_model(cfg['backbone'])
    fpn = get_fpn(cfg['fpn'])
    cls_sub = get_cls_subnet(cfg['cls_subnet'])
    reg_sub = get_reg_subnet(cfg['reg_subnet'])
    model = RetinaNet(backbone, fpn, cls_sub, reg_sub,
                      cfg['classes'], cfg['in_channels'])
    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module = Detector.load_from_checkpoint(
        'E:/projects/det_lab/saved/Frost_RetinaNet_VOC/version_2/checkpoints/last.ckpt',
        model=model)

    transformer = Transformer()

    # inference
    cls_pred, reg_pred = model_module(image_inp)
    confidences, cls_idxes, boxes = transformer(
        [image_inp, cls_pred, reg_pred])
    idxs = np.where(confidences.cpu() > 0.5)
    print(idxs)
    for i in range(idxs[0].shape[0]):
        box = boxes[idxs[0][i]]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))

    cv2.imshow('test', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    main(cfg, 'E:/VOCdevkit/VOC2012/JPEGImages/2008_000008.jpg')
