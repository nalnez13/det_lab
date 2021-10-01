from models.loss.multibox_loss import MultiBoxLoss
import pytorch_lightning as pl
from utils.module_select import get_optimizer
from module.lr_scheduler import CosineAnnealingWarmUpRestarts


class Detector(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')

        self.loss_fn = MultiBoxLoss()

    def forward(self, x):
        cls_pred, reg_pred = self.model(x)
        return cls_pred, reg_pred

    def training_step(self, batch, batch_idx):
        cls_pred, reg_pred = self.model(batch['img'])
        loss = self.loss_fn([cls_pred, reg_pred, batch])

        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cls_pred, reg_pred = self.model(batch['img'])
        loss = self.loss_fn([cls_pred, reg_pred, batch])

        self.log('val_loss', loss, logger=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        epoch_length = self.hparams.epoch_length
        optim = get_optimizer(cfg['optimizer'])(
            params=self.model.parameters(),
            **cfg['optimizer_options'])

        return {"optimizer": optim,
                "lr_scheduler": {
                    "scheduler":
                    CosineAnnealingWarmUpRestarts(
                        optim, epoch_length*4,
                        T_mult=2,
                        eta_max=cfg['optimizer_options']['lr'],
                        T_up=epoch_length,
                        gamma=0.98),
                    'interval': 'step'}
                }
