from transformers import BertModel
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
import torch
from timm.models.vision_transformer import Block
import clip


class Bert_Teacher(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.classification_head = nn.Linear(768, num_classes)
        ##self.bert = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

    def forward(self, x):
        input_ids = x["input_ids"].squeeze()
        attention_mask = x["attention_mask"].squeeze()
        token_type_ids = x["token_type_ids"].squeeze()

        x_hat = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        y_hat = self.classification_head(x_hat)
        ##y_hat = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        return y_hat


class Bert_Clinical_Teacher(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        for param in self.bert.parameters():
                param.requires_grad = False
                
        self.classification_head = nn.Linear(768, num_classes)
        ##self.bert = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

    def forward(self, x):
        input_ids = x["input_ids"].squeeze()
        attention_mask = x["attention_mask"].squeeze()
        token_type_ids = x["token_type_ids"].squeeze()

        x_hat = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        y_hat = self.classification_head(x_hat)
        ##y_hat = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        return y_hat
    

class Meta_Transformer(nn.Module):
    def __init__(self, num_classes, checkpoint_path):
        super().__init__()
        self.image_embedding = PatchEmbed().to('cuda')
        
        self.meta_encoder = load_meta_transformer(checkpoint_path).to('cuda')
        
        '''
        for param in self.meta_encoder.parameters():
                param.requires_grad = False
        '''
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1).to('cuda')
        self.classification_head = nn.Linear(768, num_classes).to('cuda')
        self.text_encoder, _ = clip.load('ViT-B/32')
        #self.text_encoder = None 
        for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        if isinstance(x, list):
            image, text = x[0], x[1]
        else:
            image = x
            text = None
        '''
        # Process the image
        image_embedding = self.image_embedding(image.to('cuda'))
        image_encoding = self.meta_encoder(image_embedding)
        image_encoding = self.global_avg_pooling(image_encoding.permute(0, 2, 1)).squeeze(dim=2)
        y_hat_img = self.classification_head(image_encoding)
        
        if text is not None:
            # Process the text
            text_embedding = get_text_embeddings(text=text, tar_dim=768, model=self.text_encoder)
            text_encoding = self.meta_encoder(text_embedding)
            y_hat_text = self.classification_head(text_encoding)
            return y_hat_img, y_hat_text.squeeze().detach()
        '''

        image_embedding = self.image_embedding(image.to('cuda'))
        if text is not None:
            text_embedding = get_text_embeddings(text=text, tar_dim=768, model=self.text_encoder)

            x = torch.cat((image_embedding, text_embedding), dim=1)

            y_hat = self.meta_encoder(x)
            y_hat = self.global_avg_pooling(y_hat.permute(0, 2, 1)).squeeze(dim=2)
            y_hat = self.classification_head(y_hat)
        else:
            y_hat = self.meta_encoder(image_embedding)
            y_hat = self.global_avg_pooling(y_hat.permute(0, 2, 1)).squeeze(dim=2)
            y_hat = self.classification_head(y_hat)


        return y_hat #y_hat_img
    
    def get_attn_maps(self, x, P=32, S_in=224, interpolation='bicubic'):
        x = self.image_embedding(x.to('cuda'))
        #x = self._pos_embed(x)
        hidden_blocks = list(self.blocks.children())[:-1]
        last_block = list(self.blocks.children())[-1]
        for blk in hidden_blocks:
            x = blk(x)
        return last_block.get_attn_maps(x, P=P, S_in=S_in, interpolation=interpolation)
    
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=32, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x    
    
def load_meta_transformer(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    
    encoder = nn.Sequential(*[
                Block(
                    dim=768,
                    num_heads=6,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU
                )
                for i in range(12)])
    
    #encoder.load_state_dict(ckpt,strict=True)
    #print('Meta Transformer initilaized with pretrained weights.')
    

    return encoder

def get_text_embeddings(text, tar_dim=768, model=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_tensor = clip.tokenize(text, truncate=True)
    
    encoding = model.encode_text(text_tensor.to('cuda'))
    encoding = zero_padding(encoding, tar_dim, device)
    encoding = encoding.unsqueeze(dim=1)
    '''
    encoding = clip.tokenize(text, truncate=True).to('cuda')
    #encoding = model.encode_text(text_tensor.to('cuda'))
    encoding = zero_padding(encoding, tar_dim, device)
    encoding = encoding.unsqueeze(dim=1)
    '''
    return encoding.to('cuda').detach()

def zero_padding(text_tensor, tar_dim, device=None):
    padding_size = tar_dim - text_tensor.shape[1]
    zero_tensor = torch.zeros((text_tensor.shape[0], padding_size), device=device)
    padded_tensor = torch.cat([text_tensor, zero_tensor], dim=1)
    return padded_tensor

