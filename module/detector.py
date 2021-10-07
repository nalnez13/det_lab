import pytorch_lightning as pl
from models.loss.multibox_loss import MultiBoxLoss
from utils.mAP_evaluate import DetectionMAP
from utils.module_select import get_optimizer

from module.lr_scheduler import CosineAnnealingWarmUpRestarts
from module.transformer import Transformer


class Detector(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.transformer = Transformer()
        self.loss_fn = MultiBoxLoss()
        self.mAP = DetectionMAP(cfg['classes'])

    def forward(self, x):
        cls_pred, reg_pred = self.model(x)
        return cls_pred, reg_pred

    def training_step(self, batch, batch_idx):
        cls_pred, reg_pred = self.model(batch['img'])
        loss = self.loss_fn([cls_pred, reg_pred, batch])

        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.mAP.reset_accumulators()

    def validation_step(self, batch, batch_idx):
        cls_pred, reg_pred = self.model(batch['img'])
        loss = self.loss_fn([cls_pred, reg_pred, batch])

        self.log('val_loss', loss, logger=True, on_epoch=True, sync_dist=True)
        confidences, cls_idxes, boxes = self.transformer(
            [batch['img'], cls_pred, reg_pred])

        for b in range(len(confidences)):
            gt_annots = batch['annot'][b]
            mask = gt_annots[:, -1] != -1
            gt_boxes = gt_annots[mask, :4]
            gt_cls = gt_annots[mask, -1]

            data = (
                boxes[b].cpu().numpy(),
                cls_idxes[b].cpu().numpy(),
                confidences[b].cpu().numpy(),
                gt_boxes.cpu().numpy(),
                gt_cls.cpu().numpy()
            )
            self.mAP.evaluate(*data)
        return loss

    def on_validation_epoch_end(self) -> None:
        ap_per_class, mAP = self.mAP.compute_map()
        self.log('val_mAP', mAP, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in ap_per_class.items():
            self.log(f'val_AP_{k}', v, on_epoch=True, sync_dist=True)

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
