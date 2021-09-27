import os
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import pytorch_lightning as pl


class TinyImageNetDataset(Dataset):
    def __init__(self, path, transforms, is_train):
        super().__init__()
        self.transforms = transforms
        self.is_train = is_train
        with open(path + '/wnids.txt', 'r') as f:
            self.label_list = f.read().splitlines()

        if is_train:
            self.data = glob.glob(path + '/train/*/images/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)
        else:
            self.data = glob.glob(path + '/val/images/*.JPEG')
            self.val_list = dict()
            with open(path + '/val/val_annotations.txt', 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]

        transformed = self.transforms(image=img)['image']
        return transformed, label


class TinyImageNet(pl.LightningDataModule):
    def __init__(self, path, workers, train_transforms, val_transforms,
                 batch_size=None):
        super().__init__()
        self.path = path
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(TinyImageNetDataset(self.path,
                                              transforms=self.train_transforms,
                                              is_train=True),
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0)

    def val_dataloader(self):
        return DataLoader(TinyImageNetDataset(self.path,
                                              transforms=self.val_transforms,
                                              is_train=False),
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0)
