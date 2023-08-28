from transformers import BertModel
import torch.nn as nn


class Bert_Teacher(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.classification_head = nn.Linear(768, num_classes)

    def forward(self, x):
        input_ids = x["input_ids"].squeeze()
        attention_mask = x["attention_mask"].squeeze()
        token_type_ids = x["token_type_ids"].squeeze()

        x_hat = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        y_hat = self.classification_head(x_hat)
      
        return y_hat