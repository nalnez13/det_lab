import argparse
import os
import random

import cv2
import numpy as np
import torch

from models.detector.retinanet import RetinaNet
from module.detector import Detector
from module.transformer import Transformer
from utils.module_select import (get_cls_subnet, get_fpn, get_model,
                                 get_reg_subnet)
from utils.utility import preprocess_input
from utils.yaml_helper import get_train_configs


def parse_names(names_file):
    names_file = os.getcwd()+names_file
    with open(names_file, 'r') as f:
        return f.read().splitlines()


def gen_random_colors(names):
    colors = [(random.randint(0, 255),
               random.randint(0, 255),
               random.randint(0, 255)) for i in range(len(names))]
    return colors


def visualize_detection(image, box, class_name, conf, color):
    x1, y1, x2, y2 = box
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color)

    caption = f'{class_name} {conf:.2f}'
    image = cv2.putText(image, caption, (x1+4, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    image = cv2.putText(image, caption, (x1+4, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return image


def main(cfg, image_name, save):
    names = parse_names(cfg['names'])
    colors = gen_random_colors(names)

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
        'E:/projects/det_lab/saved/Frost_RetinaNet_VOC/version_1/checkpoints/last.ckpt',
        model=model)

    transformer = Transformer()

    # inference
    cls_pred, reg_pred = model_module(image_inp)
    confidences, cls_idxes, boxes = transformer(
        [image_inp, cls_pred, reg_pred])
    idxs = np.where(confidences.cpu() > 0.5)
    print(idxs)
    print(cls_idxes)
    for i in range(idxs[0].shape[0]):
        box = boxes[idxs[0][i]]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        name = names[int(cls_idxes[idxs[0][i]])]
        conf = confidences[i]
        color = colors[int(cls_idxes[idxs[0][i]])]

        image = visualize_detection(image, (x1, y1, x2, y2), name, conf, color)

    cv2.imshow('test', image)
    cv2.waitKey(0)
    if save:
        cv2.imwrite('./saved/inference.png', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')
    parser.add_argument('--save', action='store_true',
                        help='Train config file')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    main(cfg, 'E:\/VOCdevkit/VOC2007/JPEGImages/000001.jpg', args.save)
