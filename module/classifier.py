import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torchmetrics import Accuracy

from utils.module_select import get_optimizer


class Classifier(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        self.top_1 = Accuracy(top_k=1)
        self.top_5 = Accuracy(top_k=5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)['pred']
        loss = F.cross_entropy(y_pred, y)

        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log('train_top1', self.top_1(y_pred, y),
                 logger=True, on_step=True, on_epoch=True)
        self.log('train_top5', self.top_5(y_pred, y),
                 logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)['pred']
        loss = F.cross_entropy(y_pred, y)
        self.log('val_loss', loss, logger=True, on_epoch=True)
        self.log('val_top1', self.top_1(y_pred, y),
                 logger=True, on_epoch=True)
        self.log('val_top5', self.top_5(y_pred, y),
                 logger=True, on_epoch=True)

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        epoch_length = self.hparams.epoch_length
        optim = get_optimizer(cfg['optimizer'])(
            params=self.model.parameters(),
            **cfg['optimizer_options'])
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optim,
            base_lr=cfg['optimizer_options']['lr']/10.,
            max_lr=cfg['optimizer_options']['lr'],
            step_size_up=epoch_length*4 if epoch_length else 2000)
        return {"optimizer": optim, "lr_scheduler": scheduler}
