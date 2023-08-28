import torch
import pytorch_lightning as pl
import torch.nn as nn

from torch import optim
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryAUROC

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from focal_loss.focal_loss import FocalLoss


class binary_train_module(pl.LightningModule):
    def __init__(self, model, lr, pos_weights):
        super().__init__()
        
        self.model = model
        self.learning_rate = lr
        
        # binary classification
        #self.output_activation = nn.Sigmoid()
        #self.loss_criterion = nn.BCELoss()
        self.pos_weights = pos_weights
        self.loss_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        self.validation_criterion = BinaryAccuracy().to(device='cuda:0')       
        self.validation_criterion_2 = BinaryAUROC().to(device='cuda:0')      
 

    def forward(self, x):
        y_hat = self.model(x)
      
        return y_hat

    def training_step(self, batch, batch_idx):
        # data
        x, y = batch
        y_hat = self.model(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()
        #y_hat = self.output_activation(y_hat)
        

        # metrics
        loss = self.loss_criterion(y_hat.float(), y.float())
        metric = self.validation_criterion(y_hat, y)
        metric_2 = self.validation_criterion_2(y_hat, y)
        
        #logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_auroc", metric_2, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss    
    
    
        
    def validation_step(self, batch, batch_idx):
        # data
        x, y = batch
        y_hat = self.model(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()
        #y_hat = self.output_activation(y_hat)
        
        # metrics
        loss = self.loss_criterion(y_hat.float(), y.float())
        metric = self.validation_criterion(y_hat, y)

        metric_2 = self.validation_criterion_2(y_hat, y)
        
        #logging
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation_accuracy", metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation_auroc", metric_2, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        

    def test_step(self, batch, batch_idx):
        # data
        x, y = batch
        y_hat = self.model(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()
        #y_hat = self.output_activation(y_hat)
        
        # metrics
        loss = self.loss_criterion(y_hat.float(), y.float())
        metric = self.validation_criterion(y_hat, y)
        metric_2 = self.validation_criterion_2(y_hat, y)

        # logging        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_auroc", metric_2, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1)

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return optimizer 