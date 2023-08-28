import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
import random

import time

from torch.utils.data import DataLoader

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from mimic_text_reports_dataset import MIMIC_TextReportsDataset
from mimic_image_dataset import train_val_test_split
from multilabel_module import multilabel_train_module
from utils import plot_train_val_loss
from utils import plot_train_val_auroc
from models import Bert_Teacher


def run(args):
    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)

    # set vars 
    data_path = args.data_path
    training_start_time = time.time()

    pathology_dict = {  0:'Atelectasis', 1:'Cardiomegaly', 2:'Consolidation'
                      , 3:'Edema', 4:'Enlarged Cardiomediastinum', 5:'Fracture'
                      , 6:'Lung Lesion', 7:'Lung Opacity', 8:'No Finding', 9:'Pleural Effusion'
                      , 10:'Pleural Other', 11:'Pneumonia', 12:'Pneumothorax', 13:'Support Devices'}

    # mimic labels
    label_file = pd.read_csv(data_path + "mimic-cxr-2.0.0-chexpert.csv")

    # train val test split
    train, test_val = train_val_test_split(label_file, 0.2)
    val, test = train_val_test_split(test_val, 0.5)

    # create data sets and loaders
    mimic_train = MIMIC_TextReportsDataset('/home/fe/baur/datasets/mimic-cxr-jpg-2.0.0-small/', train)
    mimic_val = MIMIC_TextReportsDataset('/home/fe/baur/datasets/mimic-cxr-jpg-2.0.0-small/', val)
    test_set = MIMIC_TextReportsDataset('/home/fe/baur/datasets/mimic-cxr-jpg-2.0.0-small/', test)

    print('Total of Train Images loaded: ' + str(len(mimic_train)))
    print('Total of Validation Images loaded: ' + str(len(mimic_val)))
    print('Total of Test Images loaded: ' + str(len(test_set)))

    steps_per_epoch = len(mimic_train) / args.batch_size

    # create data loaders
    train_loader = DataLoader(dataset = mimic_train
                              , batch_size = args.batch_size
                              , shuffle=True
                              , num_workers=8)
    
    val_loader = DataLoader(dataset = mimic_val
                            , batch_size = args.batch_size
                            , shuffle=False
                            , num_workers=8)
    
    
    test_loader = DataLoader(dataset = test_set
                                             , batch_size = args.batch_size
                                             , shuffle=False
                                             , num_workers=8)
    # logging
    logger = False
    if args.logging:
        logger = CSVLogger("logs", name=args.architecture)

    #### training
    torch.set_float32_matmul_precision('medium')

    ## callbacks
    # best checkpoint
    checkpoint_callback = ModelCheckpoint(save_top_k=2
                                          , monitor="validation_auroc"
                                          , mode='max')
    
    # early stop 
    early_stopping_callback = EarlyStopping(monitor="validation_auroc"
                                            , mode="max"
                                            , patience=args.early_stopping
                                            , check_on_train_epoch_end=False)
    
    # monitor lr
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(max_epochs=args.epochs
                         , logger=logger
                         , callbacks=[checkpoint_callback
                                     , early_stopping_callback
                                     , lr_monitor])
    
    print('')
    print('Logging to: ' + logger.log_dir)
    print('')
    
    network = Bert_Teacher(args.num_classes)


    model = multilabel_train_module(network
                                        , lr=args.lr
                                        , num_classes=args.num_classes
                                        , steps_per_epoch=steps_per_epoch
                                        , class_dict=pathology_dict
                                        , logging_dir=logger.log_dir
                                        , logging=args.logging
                                        , training_start_time=training_start_time
                                        )
    
    # train model
    trainer.fit(model, train_loader, val_loader)
    print('Training finished, took {:.2f}h'.format((((training_start_time - time.time()) / 60) * -1) / 60))

    checkpoint_callback.best_model_path
    
    # testing
    predictions_and_targets = {}

    trainer.test(ckpt_path=args.checkpoint, dataloaders=test_loader)
    predictions_and_targets[0] = [model.all_targets.detach().cpu().numpy()
                              , model.all_predictions.detach().cpu().numpy()]

    # plot train val metrics
    plot_train_val_loss(model.logging_dir)
    plot_train_val_auroc(model.logging_dir)

    # Save All outputs
    file_path = model.logging_dir + '/predictions_and_targets.json'
    torch.save(predictions_and_targets, file_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Run MIMIC Data')
   
    parser.add_argument('--seed', default=1337, type=int, help='global seed')
    parser.add_argument('--data_path', default='/home/fe/baur/datasets/mimic-cxr-jpg-2.0.0-small/', type=str, help='path of data set')
    parser.add_argument('--logging', default=True, type=bool, help='Enable Logger')
    parser.add_argument('--checkpoint', default='best', type=str, help='evaluate best checkpoint after training or presaved checkpoint (path in the latter case)')

    parser.add_argument('--batch_size', default=512, type=int, help='batch_size for data loaders')
    parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--architecture', default='bert', type=str, help=".")
    parser.add_argument('--lr', default=1e-5, type=float, help="Learning Rate.")
    parser.add_argument('--early_stopping', default=5, type=int, help="Early Stopping after how many epochs.")
    parser.add_argument('--num_classes', default=14, type=int, help="Number of Classes in Data Set")
    parser.add_argument('--pretrained', default=True, type=bool, help="Use Pretrained Models")
    parser.add_argument('--target_label', default='No Finding', type=str, help="Pathology to evaluate for Binary Task")
    parser.add_argument('--view_position', default='PA', type=str, help="View Position of XRay Image to evaluate.")

    args = parser.parse_args()
    run(args)




