import pytorch_lightning as pl
from models.loss.multibox_loss import MultiBoxLoss
from module.sam_optimizer import SAM
from utils.mAP_evaluate import DetectionMAP
from utils.module_select import get_optimizer

from module.lr_scheduler import CosineAnnealingWarmUpRestarts
from module.transformer import Transformer
from torch import nn


class Detector(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.transformer = Transformer()
        self.loss_fn = MultiBoxLoss(cfg)
        self.mAP = DetectionMAP(cfg['classes'])
        if cfg['optimizer'] == 'sam':
            self.automatic_optimization = False

    def forward(self, x):
        cls_pred, reg_pred = self.model(x)
        return cls_pred, reg_pred

    def training_step(self, batch, batch_idx):
        if self.hparams.cfg['optimizer'] == 'sam':
            loss, cls_loss, reg_loss = self.sam_opt_training_step(batch)
        else:
            loss, cls_loss, reg_loss = self.common_opt_training_step(batch)

        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log('train_cls_loss', cls_loss,
                 logger=True, on_step=True, on_epoch=True)
        self.log('train_reg_loss', reg_loss,
                 logger=True, on_step=True, on_epoch=True)

        return loss

    def sam_opt_training_step(self, batch):
        self.enable_running_stats(self.model)
        opt = self.optimizers()
        cls_pred, reg_pred = self.model(batch['img'])
        cls_loss, reg_loss = self.loss_fn([cls_pred, reg_pred, batch])
        loss = cls_loss + reg_loss
        self.manual_backward(loss)
        opt.first_step(zero_grad=True)

        self.disable_running_stats(self.model)
        cls_pred, reg_pred = self.model(batch['img'])
        cls_loss, reg_loss = self.loss_fn([cls_pred, reg_pred, batch])
        loss = cls_loss + reg_loss
        self.manual_backward(loss)
        opt.second_step(zero_grad=True)

        sch = self.lr_schedulers()
        sch.step()
        return loss, cls_loss, reg_loss

    def common_opt_training_step(self, batch):
        cls_pred, reg_pred = self.model(batch['img'])
        cls_loss, reg_loss = self.loss_fn([cls_pred, reg_pred, batch])
        loss = cls_loss + reg_loss
        return loss, cls_loss, reg_loss

    def on_validation_epoch_start(self):
        self.mAP.reset_accumulators()

    def validation_step(self, batch, batch_idx):
        cls_pred, reg_pred = self.model(batch['img'])
        cls_loss, reg_loss = self.loss_fn([cls_pred, reg_pred, batch])
        loss = cls_loss + reg_loss

        self.log('val_loss', loss, logger=True, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        self.log('val_cls_loss', cls_loss,
                 logger=True, on_step=True, on_epoch=True)
        self.log('val_reg_loss', reg_loss,
                 logger=True, on_step=True, on_epoch=True)
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

    def on_validation_epoch_end(self) -> None:
        ap_per_class, mAP = self.mAP.compute_map()
        self.log('val_mAP', mAP, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in ap_per_class.items():
            self.log(f'val_AP_{k}', v, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        epoch_length = self.hparams.epoch_length
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options'])

        scheduler = CosineAnnealingWarmUpRestarts(
            optim,
            epoch_length*4,
            T_mult=2,
            eta_max=cfg['optimizer_options']['lr'],
            T_up=epoch_length,
            gamma=0.96)

        return {"optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    'interval': 'step'}
                }

    @staticmethod
    def disable_running_stats(model):
        def _disable(module):
            if isinstance(module, nn.BatchNorm2d):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    @staticmethod
    def enable_running_stats(model):
        def _enable(module):
            if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        model.apply(_enable)
