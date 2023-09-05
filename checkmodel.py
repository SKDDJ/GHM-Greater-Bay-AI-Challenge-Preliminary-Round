
import torch
import utils
import os

import libs.autoencoder
import clip
from libs.clip import CLIPEmbedder
from libs.caption_decoder import CaptionDecoder
from torch.utils.data import DataLoader
from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I
import argparse
import yaml
import datetime
from pathlib import Path
from libs.data import PersonalizedBase, PromptDataset, collate_fn
from libs.uvit_multi_post_ln_v1 import UViT
import diffusers
from diffusers import DiffusionPipeline
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from pathlib import Path
import torch.nn as nn 

import tqdm

from accelerate.logging import get_logger 




def train(config):
    accelerator, device = utils.setup(config)

    train_state = utils.initialize_train_state(config, device, uvit_class=UViT)
    #print("train_state", train_state)

    train_state.nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'), False)
    
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    #print("caption_decoder", caption_decoder)
    
    
    nnet, optimizer = accelerator.prepare(train_state.nnet, train_state.optimizer)
    nnet.to(device)
    #print("nnet", nnet)
    print(nnet.state_dict())
    
    
    print(nnet.state_dict().keys())
    


    
    
  
    
    # 非Lora部分不计算梯度
    for name,param in nnet.named_parameters():
        if 'lora_adapters_ttoi' in name or 'lora_adapters_itot'  in name or 'token_embedding' in name:
            param.requires_grad = True
        else:
            param.requires_grad=False

            
    # check the nnet's parameters if they are frozen
    for name, param in nnet.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}') 

    
    # Number of trainable parameters
    print(sum(p.numel() for p in nnet.parameters() if p.requires_grad))



    print("optimizer", optimizer)
    
    
    
    lr_scheduler = train_state.lr_scheduler
    print("lr_scheduler", lr_scheduler)
    
    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    print("autoencoder", autoencoder)

    clip_text_model = CLIPEmbedder(version=config.clip_text_model, device=device)
    print("clip_text_model", clip_text_model)
    
    
    # img clip model
    clip_img_model, clip_img_model_preprocess = clip.load(config.clip_img_model, jit=False)
    
    # freeze the parameters of clip img model 
    clip_img_model.to(device).eval().requires_grad_(False)


def main():
    print("hello world")
    from configs.unidiffuserv1 import get_config
    config = get_config()
    config.ckpt_root = "/home/schengwei"
    config.workdir = "/home/schengwei"
    # print("config", config)
    
    config_name = "unidiffuserv1"
    train(config)
    print("bye world")
    
    
if __name__ == "__main__":
    main()