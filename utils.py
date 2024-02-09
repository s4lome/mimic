import torch

from tqdm import tqdm

import random

import timm

from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit 


import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from PIL import Image, ImageDraw
import torch.nn as nn


from matplotlib import gridspec

from PIL import Image, ImageDraw
import torch.nn as nn


import numpy as np
import os
import matplotlib.pyplot as plt

import traceback
import math

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




def get_attention_maps(model, image_path):
    image = Image.open(image_path)
    image = to_tensor(image).expand(3, -1, -1)
    
    attentions = model.get_attn_maps(image.unsqueeze(dim=0)).squeeze().detach()
    # Normalize the values in the tensor to be in the [0, 1] range (assuming it represents an image)
    normalized_tensor = (image - image.min()) / (image.max() - image.min())

    # Transpose the tensor to have shape 224x224x3 for compatibility with Matplotlib
    image = np.transpose(normalized_tensor, (1, 2, 0))

    num_attention_maps = attentions.shape[0]

    # Create a single row with original image on the left and attention maps on the right
    fig, axs = plt.subplots(1, num_attention_maps + 1, figsize=(3 * (num_attention_maps + 1), 7))

    # Display the original image on the leftmost subplot
    axs[0].imshow(image[..., 0], cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    for i in range(num_attention_maps):
        # Display each attention map in subsequent subplots
        axs[i + 1].imshow(image[..., 0], cmap='gray')
        axs[i + 1].imshow(attentions[i, ...], alpha=0.7)
        axs[i + 1].set_title(f'Attention Head {i}')
        axs[i + 1].axis('off')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.05)  

    # Create the 'attention_maps' folder if it doesn't exist
    if not os.path.exists('attention_maps'):
        os.makedirs('attention_maps')

    # Generate a unique filename with a timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f'attention_maps/{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}_attention.png'

    # Save the plot as an image with the unique filename
    plt.savefig(filename)
    #plt.show()
    print('')
    print('Attention Maps generated.')

    
def to_tensor(image):
    #mean = 0.4992
    #std = 0.2600
    transform_fn = Compose([Resize(256, 3)
                            , CenterCrop(224)
                            , ToTensor()
                            , Normalize(( 0.4992), (0.2600))])
    return transform_fn(image)

def pad_text_for_vit(tokens):
    pads = torch.zeros(tokens.shape[0], tokens.shape[1], (224 * 224) - tokens.shape[2]).to('cuda')
    padded_text = torch.cat((tokens, pads), dim=2)
    padded_text = padded_text.view(tokens.shape[0], 1, 224, 224)
    padded_text = padded_text.repeat(1, 3, 1, 1)

    return padded_text

'''
def get_bounding_bbox(image, image_index, annotations):
    img = image
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
        
    bounding_box = (annotations.loc[image_index,'Bbox [x']
                    , annotations.loc[image_index,'y']
                    , annotations.loc[image_index,'w']
                    , annotations.loc[image_index,'h]']
                   )
    
    outline_color = 'red'
    line_width = 8

    x, y, width, height = bounding_box
    bottom_left = (x, y)
    top_left = (x, y + height)
    bottom_right = (x + width, y)
    top_right = (x + width, y + height)

    # Draw a box with each point as a corner
    bbox = [bottom_left, top_left, top_right, bottom_right]
    draw.polygon(bbox, outline=outline_color, width=line_width)
            
    # Resize the image to 224x224
    img = img.resize((224, 224))
    
    return np.array(img), annotations.iloc[image_index,1:2], bbox
'''

def get_bounding_bbox(image, image_index, annotations):
    img = image
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
        
    bounding_box = (annotations.loc[image_index,'Bbox [x']
                    , annotations.loc[image_index,'y']
                    , annotations.loc[image_index,'w']
                    , annotations.loc[image_index,'h]']
                   )
    
    outline_color = 'red'
    line_width = 8

    x, y, width, height = bounding_box
    bottom_left = (x, y)
    top_left = (x, y + height)
    bottom_right = (x + width, y)
    top_right = (x + width, y + height)

    # Draw a box with each point as a corner
    bbox = [bottom_left, top_left, top_right, bottom_right]
    draw.polygon(bbox, outline=outline_color, width=line_width)

    # Resize the image to 224x224
    img = img.resize((224, 224))
    
    # Calculate the scaling factors
    original_width, original_height = image.size
    scale_x = 224 / original_width
    scale_y = 224 / original_height

    # Scale the bounding box coordinates
    x *= scale_x
    y *= scale_y
    width *= scale_x
    height *= scale_y

    # Update the bounding box
    adjusted_bbox = (x, y, width, height)
    
    return np.array(img), annotations.iloc[image_index, 1:2], adjusted_bbox



def get_attention_maps_with_bboxes(model, image_index):
    
    annotations = pd.read_csv('/media/baur/LaCie/CXR8/BBox_List_2017.csv')
    
    image_path = '/media/baur/LaCie/CXR8/images/all_images/' + annotations.iloc[image_index,0]
    
    image = Image.open(image_path)
    image_with_bounding_box, labels, _ = get_bounding_bbox(image, image_index, annotations)
    
    image = to_tensor(image).expand(3, -1, -1)
    
    sigmoid = nn.Sigmoid()
    prediction = sigmoid(model(image.unsqueeze(dim=0)))

    attentions = model.get_attn_maps(image.unsqueeze(dim=0)).squeeze().detach()
    # Normalize the values in the tensor to be in the [0, 1] range (assuming it represents an image)
    normalized_tensor = (image - image.min()) / (image.max() - image.min())

    # Transpose the tensor to have shape 224x224x3 for compatibility with Matplotlib
    image = np.transpose(normalized_tensor, (1, 2, 0))

    num_attention_maps = attentions.shape[0]

    # Create a subplot layout using gridspec
    num_rows = 2  # Number of rows: 1 for subplots, 1 for DataFrame
    num_cols = num_attention_maps + 1
    fig = plt.figure(figsize=(3 * (num_cols), 7))
    gs = gridspec.GridSpec(num_rows, num_cols, height_ratios=[1, 1])

    # Display the original image on the leftmost subplot
    axs = [plt.subplot(gs[0, 0])]
    axs[0].imshow(image_with_bounding_box)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Add the label as a title below the original image
    label = labels.item()
    
    prediction_dataframe = get_probabilities(prediction, label)

    for i in range(num_attention_maps):
        # Display each attention map in subsequent subplots
        axs.append(plt.subplot(gs[0, i + 1]))
        axs[i + 1].imshow(image_with_bounding_box)
        axs[i + 1].imshow(attentions[i, ...], alpha=0.4)
        axs[i + 1].set_title(f'Attention Head {i}')
        axs[i + 1].axis('off')

    # Add the prediction DataFrame as a table
    axs.append(plt.subplot(gs[1, :]))
    axs[-1].axis('off')

    # Adjust the padding and layout of the table to reduce whitespace
    table = axs[-1].table(cellText=prediction_dataframe.values
                          , colLabels=prediction_dataframe.columns
                          , loc='top', cellLoc='center')
    
    table.scale(1, 1.5) #djust the scaling to reduce whitespace

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.025, hspace=0.3)

    # Create the 'attention_maps' folder if it doesn't exist
    if not os.path.exists('attention_maps'):
        os.makedirs('attention_maps')

    '''
    # Generate a unique filename with a timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f'attention_maps/{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}_attention.png'

    # Save the plot as an image with the unique filename
    plt.savefig(filename)
    #plt.show()
    '''
    print('')
     # Display the figure
    plt.show()

    # Close the figure to release resources
    plt.close()


    '''
    # Generate a unique filename with a timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f'attention_maps/{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}_attention.png'

    # Save the plot as an image with the unique filename
    plt.savefig(filename)
    #plt.show()
    '''
    print('')
     # Display the figure
    plt.show()

    # Close the figure to release resources
    plt.close()


    #print('Attention Maps generated.')

def get_probabilities(probabilities, label):
    
    probabilities = probabilities.squeeze()
    class_labels = [0, 1, 9, 11, 12]

    # Class names dictionary
    pathology_dict = {0: 'Atelectasis', 1: 'Cardiomegaly', 2: 'Consolidation', 3: 'Edema', 4: 'Enlarged Cardiomediastinum',
                      5: 'Fracture', 6: 'Lung Lesion', 7: 'Lung Opacity', 8: 'No Finding', 9: 'Pleural Effusion',
                      10: 'Pleural Other', 11: 'Pneumonia', 12: 'Pneumothorax', 13: 'Support Devices'}

    # Class labels you want to include in the DataFrame
    class_labels = [0, 1, 9, 11, 12]

    # Create a DataFrame without the index column
    data = {
        'Finding': [label],
    }
    for label in class_labels:
        class_name = pathology_dict[label]
        probability = probabilities[label].item()  # Extract the probability as a Python float
        data[class_name] = [round(probability, 2)]  # Round the probability to two decimal places

    df = pd.DataFrame(data)
    
    # Remove the index column
    df.index = ['']

    # Display the DataFrame
    return df


'''
def compute_attention_on_pixels(attention_map, bounding_box, quantile_boundaries):
    relative_shares = []

    # Normalize the entire attention map to the range [0, 1]
    min_value = torch.min(attention_map)
    max_value = torch.max(attention_map)
    normalized_attention_map = (attention_map - min_value) / (max_value - min_value)
    
    quantiles = torch.quantile(normalized_attention_map, torch.tensor(quantile_boundaries))

    x, y, width, height = map(int, bounding_box)
    attention_bounding_box = normalized_attention_map[y:y+height, x:x+width]
 
    for quantile in quantiles:
        relative_share = torch.sum(attention_bounding_box > quantile) / (attention_bounding_box.shape[0] * attention_bounding_box.shape[1])
        relative_shares.append(relative_share)
        
    relative_shares = torch.tensor(relative_shares) * 100
    relative_shares = torch.round(relative_shares, decimals=0)

    return relative_shares, quantiles
'''

def compute_attention_on_pixels(attention_map, bounding_box, quantile_boundaries):
    relative_shares_inside = []
    relative_shares_outside = []

    # Normalize the entire attention map to the range [0, 1]
    min_value = torch.min(attention_map)
    max_value = torch.max(attention_map)
    normalized_attention_map = (attention_map - min_value) / (max_value - min_value)

    quantiles = torch.quantile(normalized_attention_map, torch.tensor(quantile_boundaries))
    #print(quantiles)
    x, y, width, height = map(int, bounding_box)
    attention_bounding_box = normalized_attention_map[y:y+height, x:x+width]

    # Calculate attention outside the bounding box
    attention_outside_bounding_box = normalized_attention_map.clone()
    attention_outside_bounding_box[y:y+height, x:x+width] = -float("inf")  # Set bounding box region to -inf

    flattened_attention_outside = attention_outside_bounding_box.view(-1)
        
    #flattened_attention_outside = flattened_attention_outside[flattened_attention_outside != -float("inf")]
    #print(torch.quantile(flattened_attention_outside, torch.tensor(quantile_boundaries)))
    
    for quantile in quantiles:
        relative_share_inside = torch.sum(attention_bounding_box > quantile) / (attention_bounding_box.shape[0] * attention_bounding_box.shape[1])
        relative_share_inside = relative_share_inside * 100
        relative_shares_inside.append(torch.round(relative_share_inside, decimals=0))



        
        relative_share_outside = torch.sum(flattened_attention_outside > quantile) / flattened_attention_outside.size(0)
        relative_share_outside = relative_share_outside * 100
        relative_shares_outside.append(torch.round(relative_share_outside, decimals=3))

    return relative_shares_inside, relative_shares_outside, quantiles


def evaluate_attention(model, quantile_boundaries):
# Define the quantile boundaries
#quantile_boundaries = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    annotations = pd.read_csv('/media/baur/Elements/nih/BBox_List_2017.csv')
    annotations.sort_values(by='Finding Label', inplace=True)
    annotations.reset_index(inplace=True)
    annotations = annotations.drop(columns=['index'])
    
    #annotations = annotations[~annotations['Finding Label'].isin(['Mass','Infiltrate','Nodule'])]
    #annotations.reset_index(inplace=True)
    #annotations = annotations.drop(columns=['index'])

    header = pd.MultiIndex.from_product([['Head_1', 'Head_2', 'Head_3'
                                          , 'Head_4', 'Head_5', 'Head_6'
                                          ], quantile_boundaries], names=['Head', 'Quantile'])
    results = pd.DataFrame(columns=header)
    results_outside_bbox = pd.DataFrame(columns=header)


    for image_index in range(0, len(annotations)):
        try:
            # get bbox
            image_path = '/media/baur/Elements/nih/all_images/' + annotations.iloc[image_index, 0]
            image = Image.open(image_path)
            img_with_bbox, labels, bbox = get_bounding_bbox(image, image_index, annotations)

            # get attentions
            image = to_tensor(image).expand(3, -1, -1)
            attentions = model.get_attn_maps(image.unsqueeze(dim=0)).squeeze().detach()
            print(attentions)
            print(attentions.shape)

            relative_shares_within_bbox_heads = []
            relative_shares_outside_bbox_heads = []

            for num_head in range(attentions.shape[0]):
                relative_shares_within_bbox_head, relative_shares_outside_bbox_head, quantiles = compute_attention_on_pixels(attentions[num_head, ...], bbox, quantile_boundaries)
                relative_shares_within_bbox_heads.append(relative_shares_within_bbox_head)
                relative_shares_outside_bbox_heads.append(relative_shares_outside_bbox_head)


            results.loc[image_index] = [value.item() for relative_shares_within_bbox_head in relative_shares_within_bbox_heads for value in relative_shares_within_bbox_head]
            results_outside_bbox.loc[image_index] = [value.item() for relative_shares_outside_bbox_head in relative_shares_outside_bbox_heads for value in relative_shares_outside_bbox_head]



        except Exception as e:
            print(f"An error occurred in iteration {image_index}: {str(e)}")
            traceback.print_exc()
            continue
        
    return results, results_outside_bbox



def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """
    Optimized version of Gumbel-Softmax sampling.
    """
    def _gen_gumbels():
        # Directly generate Gumbel noise in a stable way without recursive calls.
        gumbels = -torch.empty_like(logits).exponential_().log_()
        return gumbels
    
    gumbels = _gen_gumbels()  # Gumbel(0,1)
    y_soft = ((logits + gumbels) / tau).softmax(dim)

    if hard:
        # Use in-place operations where possible.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # Ensure the operation is done in a way that does not disrupt the computation graph.
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)