import torch

from tqdm import tqdm

import random
from PIL import ImageFilter, ImageOps

import random

import timm

from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit 


def train_val_test_split(dataframe, test_ratio):
    # split data with disjunct subject ids
    splitter = GroupShuffleSplit(test_size=test_ratio, n_splits=2, random_state = 7)
    split = splitter.split(dataframe, groups=dataframe['subject_id'])
    train_inds, test_inds = next(split)

    # first and second split
    first_split = dataframe.iloc[train_inds]
    second_split = dataframe.iloc[test_inds]

    return first_split, second_split


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
def plot_roc_curves(roc_data_dict, class_dict, num_classes, logging_dir, architecture):
    num_rows = num_classes // 4 + int(num_classes % 4 != 0)
    fig, axs = plt.subplots(nrows=num_rows, ncols=min(num_classes, 4), figsize=(16, 20))

    for i in range(num_classes):
        row_idx = i // 4
        col_idx = i % 4
        for noise_level, (all_targets, all_predictions) in roc_data_dict.items():
            fpr, tpr, thresholds = metrics.roc_curve(all_targets[:, i], all_predictions[:, i])
            roc_auc = metrics.auc(fpr, tpr)
            axs[row_idx, col_idx].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')

        axs[row_idx, col_idx].plot([0, 1], [0, 1], 'k--')
        axs[row_idx, col_idx].set_xlim([0.0, 1.0])
        axs[row_idx, col_idx].set_ylim([0.0, 1.0])
        axs[row_idx, col_idx].set_xlabel('False Positive Rate')
        axs[row_idx, col_idx].set_ylabel('True Positive Rate')
        axs[row_idx, col_idx].set_title(f'{class_dict[i]}', fontweight='bold', fontsize=14)
        axs[row_idx, col_idx].legend(loc="lower right")
        axs[row_idx, col_idx].set_aspect('equal')

    if num_classes % 4 != 0:
        for j in range(num_classes % 4, 4):
            axs[num_rows - 1, j].axis('off')

    labels = roc_data_dict.keys()
    global_legend = fig.legend(labels, title='$\\bf{Noise\ Levels}$', title_fontsize=16,
                               loc='lower right', bbox_to_anchor=(0.8, 0.05), prop={'size': 16})

    # Add global title to the entire plot
    plt.suptitle(f'ROC Curves for {architecture}', fontsize=20, fontweight='bold', y=1)

    plt.tight_layout()
    plt.savefig(logging_dir + '/' + 'auroc_curves' + '.png')
    plt.close()



def plot_train_val_loss(logging_dir):
    metrics = pd.read_csv(logging_dir + '/metrics.csv')
    
    train_loss = metrics.train_loss.dropna().reset_index().train_loss
    val_loss = metrics.validation_loss.dropna().reset_index().validation_loss
    
    # Find the epoch corresponding to the minimum validation loss
    min_val_loss_epoch = val_loss.idxmin()

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot train loss
    plt.plot(train_loss, label='Train Loss', marker='o')

    # Plot validation loss line
    plt.plot(val_loss, label='Validation Loss', marker='o')

    # Remove the marker for the minimum point of the validation curve
    plt.plot(min_val_loss_epoch
             , val_loss[min_val_loss_epoch]
             , color='red'
             , marker='s'
             , markersize=10)  

    plt.title('Train and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(len(train_loss)))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot before showing it
    plt.savefig(logging_dir + '/' + 'train_val_loss' + '.png')
    plt.close()


def plot_train_val_auroc(logging_dir):
    metrics = pd.read_csv(logging_dir + '/metrics.csv')
    
    train_auroc = metrics.train_auroc.dropna().reset_index().train_auroc
    val_auroc = metrics.validation_auroc.dropna().reset_index().validation_auroc
    
    # Find the epoch corresponding to the maximum validation AUROC
    max_val_auroc_epoch = val_auroc.idxmax()

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot train AUROC
    plt.plot(train_auroc, label='Train AUROC', marker='o')

    # Plot validation AUROC line
    plt.plot(val_auroc, label='Validation AUROC', marker='o')

    # Remove the marker for the maximum point of the validation curve
    plt.plot(max_val_auroc_epoch
             , val_auroc[max_val_auroc_epoch]
             , color='red'
             , marker='s'
             , linestyle=''
             , markersize=10)  

    plt.title('Train and Validation AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.xticks(range(len(train_auroc)))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot before showing it
    plt.savefig(logging_dir + '/' + 'train_val_auroc' + '.png')
    plt.close()


def tim_model_list(architecture):
    print(timm.list_models('*' + architecture +'*'))

   
def enable_dropout(m):
  for each_module in m.modules():
    if each_module.__class__.__name__.startswith('Drop'):
      each_module.train()


