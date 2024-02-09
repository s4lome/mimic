import argparse
import torch
import pytorch_lightning as pl
import timm 
import albumentations as A
import pandas as pd
import random
import numpy as np

import time

from torch.utils.data import DataLoader

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from albumentations.pytorch import ToTensorV2

from mimic_dataset import MIMIC_DataSet
from chexpert_dataset import CheXpert_DataSet
from chest_xray_14_dataset import Chest_XRay_14_DataSet

from utils import train_val_test_split
from binary_module import binary_train_module
from utils import plot_roc_curves
from utils import plot_train_val_loss
from utils import plot_train_val_auroc

from models import Bert_Teacher
from models import Bert_Clinical_Teacher
from models import Meta_Transformer

from multilabel_module import multilabel_train_module

from vision_transformer import vit_tiny

#from swin_models import SwinTransformer

#torch.multiprocessing.set_sharing_strategy('file_system')

def run(args):
    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)


    # set vars 
    data_path = args.data_path
    mean = 0.4992
    std = 0.2600
    training_start_time = time.time()


    pathology_dict = {  0:'Atelectasis', 1:'Cardiomegaly', 2:'Consolidation'
                      , 3:'Edema', 4:'Enlarged Cardiomediastinum', 5:'Fracture'
                      , 6:'Lung Lesion', 7:'Lung Opacity', 8:'No Finding', 9:'Pleural Effusion'
                      , 10:'Pleural Other', 11:'Pneumonia', 12:'Pneumothorax', 13:'Support Devices'}

    # transforms
    test_transforms = {}
    for i in args.noise_levels:
        test_transforms[i] = A.Compose([A.Resize(256, 256, always_apply=True)
                                        , A.CenterCrop(224, 224, always_apply=True)
                                        , A.GaussNoise(var_limit=(i), mean=0, per_channel=False, always_apply=True)
                                        , A.Normalize(mean=mean, std=std)
                                        , ToTensorV2()])


    val_transform = A.Compose([A.Resize(256, 256, always_apply=True)
                               , A.CenterCrop(224, 224, always_apply=True)
                               , A.Normalize(mean=mean, std=std)
                               , ToTensorV2()])


    train_transform = A.Compose([
        A.Resize(256, 256, always_apply=True),
        A.CenterCrop(224, 224, always_apply=True),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Cutout(num_holes=4, max_h_size=32, max_w_size=32, fill_value=0, always_apply=False, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
])

    # mimic labels
    label_file = pd.read_csv(data_path + "mimic-cxr-2.0.0-chexpert.csv")

    # train val test split
    train, test_val = train_val_test_split(label_file, 0.2)
    val, test = train_val_test_split(test_val, 0.5)

    if args.architecture=='meta_transformer':
        tokenize=False
    else:
        tokenize=True


    # create data sets and loaders
    mimic_train = MIMIC_DataSet(args.data_path, train, train_transform, args.task, args.target_label, args.view_position, tokenize=tokenize)
    mimic_val = MIMIC_DataSet(args.data_path, val, val_transform, args.task, args.target_label, args.view_position, tokenize=tokenize)
    


    test_sets = {}
    if args.ood:

        for i in args.noise_levels:
            annotations = pd.read_csv('/media/baur/LaCie/CXR8/Data_Entry_2017_v2020.csv')
            #annotations = annotations[annotations['View Position'] == 'PA']
            #annotations = annotations.sample(frac=0.05)
            with open('/media/baur/LaCie/CXR8/test_list.txt', 'r') as file:
                test_set = file.read().splitlines()
            annotations = annotations[annotations['Image Index'].isin(test_set)]    
            #test_sets[i] = CheXpert_DataSet('/home/fe/baur/datasets', pathology_dict=pathology_dict, transforms=test_transforms[i])
            test_sets[i] = Chest_XRay_14_DataSet('/media/baur/LaCie/CXR8/images/all_images', annotations, pathology_dict, transforms=test_transforms[i])

            
    else:
        for i in args.noise_levels:
            test_sets[i] = MIMIC_DataSet(args.data_path, test, test_transforms[i], args.task, args.target_label, args.view_position, tokenize=tokenize)


    print('Total of Train Images loaded: ' + str(len(mimic_train)))
    print('Total of Validation Images loaded: ' + str(len(mimic_val)))
    print('Total of Test Images loaded: ' + str(len(test_sets[0])))

    steps_per_epoch = len(mimic_train) / args.batch_size

    # create data loaders
    train_loader = DataLoader(dataset = mimic_train
                              , batch_size = args.batch_size
                              , shuffle=False
                              , num_workers=8)
    
    val_loader = DataLoader(dataset = mimic_val
                            , batch_size = args.batch_size
                            , shuffle=False
                            , num_workers=8)
    


    
    test_loaders = {}
    for i in args.noise_levels:
        test_loaders[i] = DataLoader(dataset = test_sets[i]
                                             , batch_size = args.batch_size
                                             , shuffle=False
                                             , num_workers=8)
    # logging
    logger = False
    if args.logging:
        logger = CSVLogger("logs", name=args.architecture)

    bert=None
    if args.priviliged_knowledge:
        # load fine tuned bert teacher
        bert = Bert_Teacher(args.num_classes)
        #bert = Bert_Clinical_Teacher(args.num_classes)

        teacher_module = multilabel_train_module(bert
                                            , lr=1e-5
                                            , num_classes=14
                                            , steps_per_epoch=1000
                                            , class_dict={}
                                            , logging_dir=''
                                            , logging=True
                                            , training_start_time=''
                                            #, batch_size = args.batch_size
                                            )
        
        checkpoint = torch.load(args.bert_checkpoint)
        state_dict = checkpoint['state_dict']
        teacher_module.load_state_dict(state_dict)
        bert = teacher_module.model

        print('Bert Teacher initialized.')

    # create model
    if args.architecture == 'meta_transformer':
        network = Meta_Transformer(args.num_classes, '/home/fe/baur/Downloads/Meta-Transformer_base_patch16_encoder (1).pth')
    else:
        '''
        network = timm.create_model(args.architecture
                                    , num_classes=args.num_classes
                                    , drop_rate = args.dropout_rate
                                    #, pos_drop_rate = args.dropout_rate
                                    #, proj_drop_rate = args.dropout_rate
                                    , attn_drop_rate = args.dropout_rate
                                    , drop_path_rate = args.dropout_rate
                                    #, patch_size=16, embed_dim=384
                                    , pretrained=args.pretrained
                                    )
        '''
        #network = SwinTransformer(num_classes=14)
        network = vit_tiny(num_classes=14, drop_path_rate=args.dropout_rate, drop_rate=args.dropout_rate)
    #### training
    torch.set_float32_matmul_precision('high')

    ## callbacks
    # best checkpoint
    checkpoint_callback = ModelCheckpoint(save_top_k=1
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
    
    if args.task == 'binary':
        model = binary_train_module(network
                                    , lr=args.lr
                                    , pos_weights=None)
   
    if args.task == 'multilabel':
        model = multilabel_train_module(network
                                        , teacher=bert
                                        , imitation=args.imitation
                                        , temperature=args.temperature
                                        , lr=args.lr
                                        , num_classes=args.num_classes
                                        , steps_per_epoch=steps_per_epoch
                                        , class_dict=pathology_dict
                                        , logging_dir=logger.log_dir
                                        , logging=args.logging
                                        , training_start_time=training_start_time
                                        , mode = args.mode
                                        )
    # load checkpoint
    #checkpoint = torch.load("/home/fe/baur/wd/projects/mimic/logs/vit_large_patch16_224/version_2/checkpoints/epoch=1-step=16396.ckpt")
    #model.load_state_dict(checkpoint["state_dict"])
    
    # train model
    trainer.fit(model, train_loader, val_loader)
    print('Training finished, took {:.2f}h'.format((((training_start_time - time.time()) / 60) * -1) / 60))

    checkpoint_callback.best_model_path
    
    # testing
    predictions_and_targets = {}

    for i in test_loaders:
        trainer.test(ckpt_path=args.checkpoint, dataloaders=test_loaders[i])
        predictions_and_targets[i] = [model.all_targets.detach().cpu().numpy()
                                      , model.all_predictions.detach().cpu().numpy()]

    # plot auroc curves
    plot_roc_curves(predictions_and_targets
                    , model.class_dict
                    , model.num_classes
                    , model.logging_dir
                    , args.architecture)
    
    try:
        # plot train val metrics
        plot_train_val_loss(model.logging_dir)
        plot_train_val_auroc(model.logging_dir)
    except Exception as e:
        print('No Training plots created.')
    
    # Save All outputs
    file_path = model.logging_dir + '/predictions_and_targets.json'
    torch.save(predictions_and_targets, file_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Run MIMIC Data')
   
    parser.add_argument('--seed', default=1337, type=int, help='global seed')
    parser.add_argument('--data_path', default='/home/fe/baur/datasets/mimic-cxr-jpg-2.0.0-small/', type=str, help='path of data set')
    parser.add_argument('--logging', default=True, type=bool, help='Enable Logger')
    parser.add_argument('--checkpoint', default='best', type=str, help='evaluate best checkpoint after training or presaved checkpoint (path in the latter case)')
    parser.add_argument('--priviliged_knowledge', default=False, type=bool, help='Train with Privilegded Knowledge')
    parser.add_argument('--ood', default=False, type=bool, help='Evaluate on OOD Data')

    parser.add_argument('--bert_checkpoint', default='/home/fe/baur/wd/projects/mimic/logs/bert/no_labels/checkpoints/epoch=0-step=29719.ckpt', type=str, help='path to .ckpt of fine tuned bert model.')

    parser.add_argument('--batch_size', default=512, type=int, help='batch_size for data loaders')
    parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--architecture', default='vit', type=str, help="Model Architecture. One of ['vit', 'vit_small', 'swin_tiny', 'swin_small','vgg', 'densenet', 'resnet'].")
    parser.add_argument('--lr', default=1e-5, type=float, help="Learning Rate.")
    parser.add_argument('--early_stopping', default=5, type=int, help="Early Stopping after how many epochs.")
    parser.add_argument('--task', default='binary', type=str, help="Task to evaluate. One of ['binary', 'multilabel'].")
    parser.add_argument('--num_classes', default=14, type=int, help="Number of Classes in Data Set")
    parser.add_argument('--pretrained', default=True, type=bool, help="Use Pretrained Models")
    parser.add_argument('--target_label', default='No Finding', type=str, help="Pathology to evaluate for Binary Task")
    parser.add_argument('--view_position', default='PA', type=str, help="View Position of XRay Image to evaluate.")

    parser.add_argument('--dropout_rate', default=0, type=float, help="Dropout Rate. Same for all Dropout types of Transformer")
    parser.add_argument('--enable_mc_dropout', default=False, type=bool, help="Wheter to use MC Dropout for testing")

    parser.add_argument('--noise_levels', default = 0, nargs="+", type=int)

    parser.add_argument('--imitation', default=0, type=float, help="Imitation Rate for Priviliged Knowledge Learning")
    parser.add_argument('--temperature', default=1, type=float, help="Temperature for Privilged Knowledge Learning")
    parser.add_argument('--mode', default='standard', type=str, help="Train Step mode (standard, dual, meta)")

    
    args = parser.parse_args()

     # Print all parsed arguments dynamically
    print("Arguments:")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    run(args)