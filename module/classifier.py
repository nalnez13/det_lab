import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from utils.module_select import get_optimizer


class Classifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)['pred']
        loss = F.cross_entropy(y_pred, y)

        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        return get_optimizer(cfg['optimizer'])(
            params=self.model.parameters(),
            **cfg['optimizer_options'])
