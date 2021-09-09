import argparse

from utils.utility import make_model_name

import albumentations
import albumentations.pytorch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import tiny_imagenet
from utils.module_select import get_model
from utils.yaml_helper import get_train_configs
from module.classifier import Classifier


def train(cfg):
    transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)

    model = get_model(cfg['model'])(in_channels=3, classes=cfg['classes'])
    model_module = Classifier(model, cfg=cfg)
    data_module = tiny_imagenet.TinyImageNet(
        path=cfg['data_path'], workers=cfg['workers'],
        train_transforms=transforms, val_transforms=transforms,
        batch_size=cfg['batch_size'])

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'],
                                 make_model_name(cfg)),
        gpus=cfg['gpus'],
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=[],
        **cfg['trainer_options'])
    trainer.fit(model_module, data_module)


# def find_optimal_lr(cfg):
#     transforms = albumentations.Compose([
#         albumentations.HorizontalFlip(p=0.5),
#         albumentations.VerticalFlip(p=0.5),
#         albumentations.Normalize(0, 1),
#         albumentations.pytorch.ToTensorV2(),
#     ],)

#     model = get_model(cfg['model'])(in_channels=3, classes=cfg['classes'])
#     model_module = Classifier(model, cfg=cfg, lr=cfg['lr'])
#     data_module = tiny_imagenet.TinyImageNet(
#         path=cfg['data_path'], workers=cfg['workers'],
#         train_transforms=transforms, val_transforms=transforms,
#         batch_size=cfg['batch_size'])

#     trainer = pl.Trainer(
#         auto_lr_find="lr",
#         gpus=cfg['gpus'],
#         accelerator='ddp',
#         plugins=DDPPlugin(find_unused_parameters=False))

#     # trainer.tune(model_module, data_module)
#     lr_find = trainer.tuner.lr_find(model_module, data_module)
#     fig = lr_find.plot(suggest=True)
#     print('found lr: ', lr_find.suggestion())
#     fig.savefig('./lr_find.png')
#     print('done?')
#     trainer._on_expection
#     exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')
    # parser.add_argument('--find_lr', required=False, action='store_true',
    #                     help='find optimal learning rate')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    # find_lr = args.find_lr

    # if find_lr:
    #     find_optimal_lr(cfg)
    # else:
    train(cfg)
