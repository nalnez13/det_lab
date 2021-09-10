from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np


class YoloDataset(Dataset):
    def __init__(self, path, transforms):
        super().__init__()
        self.imgs = []

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        annot = self.load_annotations(img_file)

        return super().__getitem__(index)

    def load_annotations(self, img_file):
        annotations_file = img_file.replace('.jpg', '.txt')
        img_annotations = np.zeros((0, 5))
        with open(annotations_file, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                cid, cx, cy, w, h = map(float, annot.split(' '))
                x1 = cx-w/2
                y1 = cy-h/2
                x2 = cx+w/2
                y2 = cy+h/2
                annotation = np.array([x1, y1, x2, y2, cid])
                img_annotations = np.append(
                    img_annotations, annotation, axis=0)
        return img_annotations
