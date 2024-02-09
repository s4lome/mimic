
import pandas as pd
from torch.utils.data import Dataset
import cv2
import torch

import numpy as np
import torch



class Chest_XRay_14_DataSet(Dataset):
    def __init__(self, data_path, annotation_file, pathology_dict, transforms=None):
        
        self.data_path = data_path
        self.annotations_file = preprocess_annotations(annotation_file, pathology_dict)
        self.pathology_dict = pathology_dict
        self.transform = transforms

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        # vars for img path
        image_id = self.annotations_file["Image Index"].iloc[index]
        img_path = self.data_path + '/' + image_id
        
        #image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #image = Image.open(img_path).convert('L')
        #image = np.array(image)
        #image = np.repeat(image, 3, axis=0)
        image = cv2.imread(img_path)
        label = torch.tensor(self.annotations_file.iloc[index,1],dtype=torch.long)
        
        # apply transforms
        if self.transform:
            #image = self.transform(image)
            image = self.transform(image=image)["image"]
            
        
        return [image], label
    
    
def preprocess_annotations(annotations, pathology_dict):
    annotations['split_labels'] = annotations['Finding Labels'].str.split('|')
    pathology_dict[9] = 'Effusion'
    for label in pathology_dict.values():
        annotations[label] = 0
        annotations[label] = annotations.apply(lambda row: 1 if label in row['split_labels'] else 0, axis=1)
    
    annotations['labels'] = annotations.iloc[:,-14:].values.tolist()
    annotations = annotations[['Image Index', 'labels']]
    return annotations