# import os
# import torch
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.randn((200000, 300, 200, 20), device=device)

import os

import torch

from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver

from libs.uvit_multi_post_ln_v1 import UViT
from peft import inject_adapter_in_model, LoraConfig,get_peft_model

lora_config = LoraConfig(
   inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1,target_modules=["qkv","fc1","fc2","proj","text_embed","clip_img_embed"]
)


from configs.sample_config import get_config


config = get_config()

config.lora_path = os.path.join("model_output/girl2_copy", "lora.pt.tmp",'lora.pt')

device = "cuda"

# init models
nnet = UViT(**config.nnet)

print(f'load nnet from {config.lora_path}')

nnet.load_state_dict(torch.load("/home/wuyujia/competition/models/uvit_v1.pth", map_location='cpu'), False)
print(nnet)

nnet = get_peft_model(nnet,lora_config)

nnet.load_state_dict(torch.load(config.lora_path, map_location='cpu'), False)

