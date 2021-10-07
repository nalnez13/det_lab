from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
import pytorch_lightning as pl
from dataset.detection.utils import collater


class YoloDataset(Dataset):
    def __init__(self, transforms, path=None, files_list=None):
        super().__init__()
        assert path is not None or files_list is not None

        if path:
            self.imgs = glob.glob(path+'/*.jpg')
        if files_list:
            with open(files_list, 'r') as f:
                self.imgs = f.read().splitlines()
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file = self.imgs[index]
        img = cv2.imread(img_file)
        boxes = self.load_annotations(img_file, img.shape)
        transformed = self.transforms(image=img, bboxes=boxes)
        return transformed

    def load_annotations(self, img_file, img_shape):
        img_h, img_w, _ = img_shape
        annotations_file = img_file.replace('.jpg', '.txt')
        boxes = np.zeros((0, 5))
        with open(annotations_file, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                cid, cx, cy, w, h = map(float, annot.split(' '))
                x1 = (cx-w/2)*img_w
                y1 = (cy-h/2)*img_h
                w *= img_w
                h *= img_h
                annotation = np.array([[x1, y1, w, h, cid]])
                boxes = np.append(boxes, annotation, axis=0)

        return boxes


class YoloFormat(pl.LightningDataModule):
    def __init__(self, train_list, val_list, workers, train_transforms,
                 val_transforms,
                 batch_size=None):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(YoloDataset(
            transforms=self.train_transforms,
            files_list=self.train_list),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=collater)

    def val_dataloader(self):
        return DataLoader(YoloDataset(
            transforms=self.val_transforms,
            files_list=self.val_list),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=collater)


if __name__ == '__main__':
    """
    Data loader 테스트 코드
    python -m dataset.detection.yolo_format
    """
    import albumentations
    import albumentations.pytorch
    from dataset.detection.utils import visualize

    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Posterize(),
        albumentations.RandomGamma(),
        albumentations.Equalize(),
        albumentations.HueSaturationValue(),
        albumentations.RandomBrightnessContrast(),
        albumentations.ColorJitter(),
        albumentations.ShiftScaleRotate(),
        albumentations.Resize(300, 300, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    loader = DataLoader(YoloDataset(
        transforms=train_transforms, files_list='e:/voc_val.txt'),
        batch_size=16, shuffle=True, collate_fn=collater)

    for batch, sample in enumerate(loader):
        imgs = sample['img']
        annots = sample['annot']
        visualize(imgs, annots)
