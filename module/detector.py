import argparse

from utils.utility import make_model_name

import albumentations
import albumentations.pytorch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dataset import tiny_imagenet
from utils.module_select import get_model
from utils.yaml_helper import get_train_configs
from module.classifier import Classifier


def train(cfg):
    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Affine(),
        albumentations.ColorJitter(),
        albumentations.RandomBrightnessContrast(),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)

    valid_transform = albumentations.Compose([
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)
    data_module = tiny_imagenet.TinyImageNet(
        path=cfg['data_path'], workers=cfg['workers'],
        train_transforms=train_transforms, val_transforms=valid_transform,
        batch_size=cfg['batch_size'])

    model = get_model(cfg['model'])(in_channels=3, classes=cfg['classes'])
    model_module = Classifier(
        model, cfg=cfg, epoch_length=data_module.train_dataloader().__len__())

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(monitor='val_loss', save_last=True,
                        every_n_epochs=cfg['save_freq'])
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'],
                                 make_model_name(cfg)),
        gpus=cfg['gpus'],
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=callbacks,
        **cfg['trainer_options'])
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)

    train(cfg)