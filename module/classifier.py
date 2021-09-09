import pytorch_lightning as pl
import torch.nn.functional as F

from torchmetrics import Accuracy

from utils.module_select import get_optimizer


class Classifier(pl.LightningModule):
    def __init__(self, model, cfg, lr):
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
        self.log('val_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log('val_top1', self.top_1(y_pred, y),
                 logger=True, on_step=True, on_epoch=True)
        self.log('val_top5', self.top_5(y_pred, y),
                 logger=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        lr = self.hparams.lr
        return get_optimizer(cfg['optimizer'])(
            params=self.model.parameters(),
            lr=lr,
            **cfg['optimizer_options'])
