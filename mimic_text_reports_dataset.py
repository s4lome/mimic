import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re

class MIMIC_TextReportsDataset(Dataset):
    def __init__(self, path, label_file):     
        self.label_file = label_file
        self.path=path

        self.annotations_file = preprocces_annotations(self)

        self.labels = self.annotations_file.iloc[:,2]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index): 
        # relevant identifiers
        study_id = self.annotations_file["study_id"].iloc[index]
        subject_id = self.annotations_file["subject_id"].iloc[index]
        
        # path
        text_path = self.path + '/files/' + "p" + str(subject_id)[:2] + "/p" + str(subject_id) + "/s" + str(study_id) + ".txt"
        
        # get text
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read()

       # Preprocess text
        text = preprocess_text(text)
        #print(text)
        # tokenize 
        token = self.tokenizer(text,padding='max_length', max_length = 512, 
                           truncation=True, return_tensors="pt")
        
        # get labels
        label = torch.tensor(self.labels[index]).long()
        
        return [token], label
    
def preprocces_annotations(self):
    #label_file = pd.read_csv(path + "mimic-cxr-2.0.0-chexpert.csv")
    label_file = self.label_file
    #process labels
    label_file = label_file.fillna(0)
    label_file = label_file[~(label_file == -1).any(axis=1)]

    # read meta data file (377,110 entries)
    meta_data = pd.read_csv(self.path + "mimic-cxr-2.0.0-metadata.csv")

    # right join
    annotations = pd.merge(meta_data, label_file, how='right')
    annotations=annotations[annotations['ViewPosition']=='PA']
    
    # concat values to list
    annotations['labels'] = annotations.iloc[:,12:].values.tolist()
    
    #only relevant cols
    annotations = annotations[['subject_id', 'study_id','labels']]
    #annotations = annotations.head(1024)
    
    # reset index
    annotations = annotations.reset_index().drop('index', axis=1)

    return annotations

def preprocess_text(text):
    # Define words to omit (in lowercase)
    words_to_omit = ['pleural effusion', 'effusion', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusions',
    'Pleural', 'Pneumonia', 'Pneumothorax', 'Support Devices']  # Add more words if needed

    # Use case-insensitive regular expression to find and replace words to omit
    for word in words_to_omit:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        text = pattern.sub('_', text)

    return text

