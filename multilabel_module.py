import torch
import pytorch_lightning as pl
import torch.nn as nn
import time 

from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification import BinaryAUROC

from models import Meta_Transformer

class multilabel_train_module(pl.LightningModule):
    def __init__(self
                 , model=None
                 , teacher=None
                 , imitation=0
                 , temperature=0
                 , lr=1e-5
                 , num_classes=14
                 , steps_per_epoch=None
                 , class_dict={}
                 , logging_dir=''
                 , logging=True
                 , training_start_time=''):
        
        super().__init__()
        
        self.model = model
        self.teacher = teacher 
        self.imitation = 0
        if self.teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
                self.imitation = imitation
        self.temperature = temperature

        self.num_classes = num_classes
        self.learning_rate = lr
        self.steps_per_epoch = steps_per_epoch
        self.class_dict = class_dict
        self.logging_dir = logging_dir
        self.logging = logging
        self.training_start_time = training_start_time

        # init global storage
        self.train_step_outputs = []
        self.train_step_targets = []      
        self.validation_step_outputs = []
        self.validation_step_targets = []       
        self.test_step_outputs = []
        self.test_step_targets = []
        
        # multi label classification
        self.output_activation = nn.Sigmoid()
        self.loss_criterion = nn.BCEWithLogitsLoss()
        self.auroc_global = MultilabelAUROC(num_labels=self.num_classes, average="macro", thresholds=None).to(device='cuda:0')
        self.auroc_single_class = BinaryAUROC().to(device='cuda:0')

    def forward(self, x):
        y_hat = self.model(x)
      
        return y_hat

    def training_step(self, batch, batch_idx):
        # data
        image_and_text , y = batch

        if isinstance(self.model, Meta_Transformer):
            loss = meta_transformer_train_step(self, image_and_text, y)

        else:
            loss = privileged_knowlegde_train_step(self, image_and_text, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=self.logging, batch_size=y.shape[0])

        return loss    
    
    def on_train_epoch_end(self):
        # concat all steps
        all_predictions = torch.cat(self.train_step_outputs, dim=0)
        all_targets = torch.cat(self.train_step_targets, dim=0)
            
        # activation
        all_predictions = self.output_activation(all_predictions)

        # global auroc
        auroc_global = self.auroc_global(all_predictions, all_targets)
  
        self.log("train_auroc", auroc_global, on_step=False, on_epoch=True, prog_bar=True, logger=self.logging)
        self.log("training_time", (((self.training_start_time - time.time()) / 60) * -1) / 60,
                  on_step=False, on_epoch=True, prog_bar=True, logger=self.logging)


        # free memory
        self.train_step_outputs.clear()
        self.train_step_targets.clear()
    
    def validation_step(self, batch, batch_idx):
        # data
        x, y = batch
        
        x = x[0]

        y_hat = self.model(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()

        # add to global
        self.validation_step_outputs.append(y_hat.detach())
        self.validation_step_targets.append(y.detach())
        
        # loss
        loss = self.loss_criterion(y_hat.float(), y.float())

        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=self.logging, batch_size=y.shape[0])

    def on_validation_epoch_end(self):
        # concat all steps
        all_predictions = torch.cat(self.validation_step_outputs, dim=0)
        all_targets = torch.cat(self.validation_step_targets, dim=0)
            
        # activation
        all_predictions = self.output_activation(all_predictions)

        # global auroc
        auroc_global = self.auroc_global(all_predictions, all_targets)
  
        self.log("validation_auroc", auroc_global, on_step=False, on_epoch=True, prog_bar=True, logger=self.logging)

        # free memory
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

    def test_step(self, batch, batch_idx):      
        # data
        x, y = batch
        
        x = x[0]

        y_hat = self.model(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()

        # add to global
        self.test_step_outputs.append(y_hat.detach())
        self.test_step_targets.append(y.detach())
        
        # loss
        loss = self.loss_criterion(y_hat.float(), y.float())
        self.log("Test BCE Loss: ", loss, on_step=False, on_epoch=True, prog_bar=True, logger=self.logging, batch_size=y.shape[0])
  
    def on_test_epoch_end(self):
        # concat all steps
        self.all_predictions = torch.cat(self.test_step_outputs, dim=0)
        self.all_targets = torch.cat(self.test_step_targets, dim=0)
        
        # activation
        self.all_predictions = self.output_activation(self.all_predictions)

        # global auroc
        auroc_global = self.auroc_global(self.all_predictions, self.all_targets)
        self.log("auroc Global: ", auroc_global, on_step=False, on_epoch=True, prog_bar=True, logger=self.logging)

        # aurocs per class
        for i in range(0,self.num_classes):
            #claculate class auroc
            class_auroc = self.auroc_single_class(torch.index_select(self.all_predictions
                                                                    , 1
                                                                    , torch.tensor(i).to(device='cuda:0'))
                                                ,torch.index_select(self.all_targets
                                                                    , 1
                                                                    , torch.tensor(i).to(device='cuda:0'))
                                                )
            # log class
            self.log(('auroc ' + self.class_dict[i] +':'), class_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=self.logging)

        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1)
        #scheduler = CosineAnnealingLR(optimizer,
        #                      T_max = self.steps_per_epoch, # Maximum number of iterations.
        #                     eta_min = 1e-7) # Minimum learning rate.
        return optimizer#, [scheduler]
    

def privileged_knowlegde_train_step(self, image_and_text, y):
        x = image_and_text[0]
        
        y_hat = self.model(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()

        if self.teacher:
            pk = image_and_text[1]
            soft_labels = self.teacher(pk)
            soft_labels = soft_labels / self.temperature
            soft_labels = self.output_activation(soft_labels)
          # soft_labels = (soft_labels >= 0.5).to(torch.float32)

          #   y_hat = (1 - self.imitation) * y_hat + self.imitation * soft_labels

        # add to global
        self.train_step_outputs.append(y_hat.detach())
        self.train_step_targets.append(y.detach())
        
        #loss = self.loss_criterion(y_hat.float(), y.float())

        # loss
        loss = (1 - self.imitation) * self.loss_criterion(y_hat.float(), y.float())
        
        if self.teacher:
            loss += self.imitation * self.loss_criterion(y_hat.float(), soft_labels)
        
        return loss


def meta_transformer_train_step(self, image_and_text, y):
    
    y_hat_image, y_hat_text = self.model(image_and_text)
    
    loss_image = self.loss_criterion(y_hat_image.float(), y.float())
    loss_text = self.loss_criterion(y_hat_text.float(), y.float())
    
    y_hat = (y_hat_image + y_hat_text) / 2
    self.train_step_outputs.append(y_hat.detach())
    self.train_step_targets.append(y.detach())

    loss = loss_image + loss_text

    return loss