import argparse
import timm
import torch

from multilabel_module import multilabel_train_module
from utils import get_attention_maps

def run(args):
    
    network = timm.create_model(args.architecture
                                    , num_classes=14
                                    , attn_drop_rate = 0
                                    , drop_path_rate = 0
                                    , pretrained=True)
    
    model = multilabel_train_module(network)
    model.eval()

    #load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    network=model.model

    get_attention_maps(network,args.image_path)
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Attention')

    parser.add_argument('--architecture', default='vit_base_patch16_224', type=str, help="Model Architecture.")
    parser.add_argument('--image_path', default='/home/fe/baur/datasets/mimic-cxr-jpg-2.0.0-small/files_small/p10/p10001217/s52067803/a917c883-720a5bbf-02c84fc6-98ad00ac-c562ff80.jpg'
                        , type=str, help='Path to the image for which you want to generate attention maps.')
    parser.add_argument('--checkpoint_path', default='/home/fe/baur/wd/projects/mimic/logs/vit_base_patch16_224/no dropout/checkpoints/epoch=1-step=4100.ckpt', 
                        type=str, help='Path to the model checkpoint for generating attention maps.')

    args = parser.parse_args()

     # Print all parsed arguments dynamically
    print("Arguments:")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    run(args)


