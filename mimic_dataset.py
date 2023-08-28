import pandas as pd
import torch
import cv2

from torch.utils.data import Dataset

from sklearn.model_selection import GroupShuffleSplit 


from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset


class MIMIC_DataSet(Dataset):
    def __init__(self, path, label_file, transform, task, target_label, view_position):
        self.path=path
        self.task=task
        self.target_label = target_label
        self.label_file = label_file
        self.view_position = view_position
        self.annotations_file = preprocces_annotations(self)      
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        # vars for img path
        study_id = self.annotations_file["study_id"].iloc[index]
        subject_id = self.annotations_file["subject_id"].iloc[index]
        dicom_id = self.annotations_file["dicom_id"].iloc[index]

        # path has the format of e.g.: 
        # path/files_small/p10/p10000032/dicom_id
        img_path = self.path + "files_small/" + "p" + str(subject_id)[:2] + "/p" + str(subject_id) + "/s" + str(study_id) + "/" + dicom_id + ".jpg"
        image = cv2.imread(img_path)
        
        # get labels, 4th col of annotations file contains labels
        label = torch.tensor(self.annotations_file.iloc[index,3],dtype=torch.long)
        
        # apply transforms
        if self.transform:
            #image = self.transform(image)
            image = self.transform(image=image)["image"]
            
        # get text report
        # path
        text_path = self.path + '/files/' + "p" + str(subject_id)[:2] + "/p" + str(subject_id) + "/s" + str(study_id) + ".txt"
        
        # get text
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read()
       
        # tokenize 
        token = self.tokenizer(text,padding='max_length', max_length = 512, 
                           truncation=True, return_tensors="pt")
           
        
        return (image,token), label
    
def preprocces_annotations(self):
    # read label file (227,827 entries)
    #label_file = pd.read_csv(path + "mimic-cxr-2.0.0-chexpert.csv")
    
    #process labels
    label_file = self.label_file.fillna(0)

    label_file = label_file[~(label_file == -1).any(axis=1)]
    #label_file = label_file.loc[~(label_file['Atelectasis'] == 1)]
    
    if self.task == 'binary':
        label_file = label_file[['subject_id', 'study_id', self.target_label]]

    # read meta data file (377,110 entries)
    meta_data = pd.read_csv(self.path + "mimic-cxr-2.0.0-metadata.csv")

    # right join
    annotations = pd.merge(meta_data, label_file, how='right')
    annotations=annotations[annotations['ViewPosition']==self.view_position]
    
    # concat values to list
    annotations['labels'] = annotations.iloc[:,12:].values.tolist()
    
    #rest indices
    annotations = annotations.reset_index().drop('index', axis=1)

    #only relevant cols
    annotations = annotations[['dicom_id','subject_id', 'study_id','labels']]
    #annotations = annotations.head(1024)

    return annotations

