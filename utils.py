import torch
import torch.nn as nn
import numpy as np
import os
from absl import logging
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from libs.clip import FrozenCLIPEmbedder
import itertools
from libs.clip import CLIPEmbedder
from peft import inject_adapter_in_model, LoraConfig,get_peft_model
# lora_config = LoraConfig(
#    inference_mode=False, r=128, lora_alpha=90, lora_dropout=0.1,target_modules=["qkv","fc1","fc2","proj","to_out","to_q","to_k","to_v","text_embed","clip_img_embed"]
# )
lora_config = LoraConfig(
   inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1,target_modules=["qkv","fc1","fc2","proj","text_embed","clip_img_embed"]
)
# lora_config = LoraConfig(
#    inference_mode=False, r=128, lora_alpha=90, lora_dropout=0.1,target_modules=["qkv","fc1","fc2","proj","text_embed","clip_img_embed"]
# )
# lora_config = LoraConfig(
#    inference_mode=False, r=128, lora_alpha=64, lora_dropout=0.1,target_modules=["qkv","to_out","to_q","to_k","to_v","text_embed","clip_img_embed"]
# )#94,838,784
# lora_config = LoraConfig(
#    inference_mode=False, r=128, lora_alpha=64, lora_dropout=0.1,target_modules=["qkv","text_embed","clip_img_embed"]
# )

def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem

def get_data_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--data='):
            return Path(argv[i].split('=')[-1]).stem
        

def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})



def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 20, T_mult=2, **kwargs)
        # return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    """ 
    这个函数是用于实现模型参数的指数移动平均（Exponential Moving Average，EMA）的。具体而言，它将源模型的参数按照一定的比例rate融合到目标模型的参数中。

函数的输入参数包括：

model_dest: 目标模型
model_src: 源模型
rate: 融合比例，通常取值在[0, 1]之间
函数具体实现的步骤如下：

将源模型的参数按照名称转化为字典param_dict_src。
遍历目标模型的参数p_dest，对于每个参数，找到对应名称的源模型参数p_src。
利用assert语句确保p_src和p_dest不是同一个对象。
将p_dest的数值乘以rate后加上(1-rate)倍的p_src数值，得到融合后的结果，并将结果赋值给p_dest。
这个函数的作用是在训练神经网络时，通过融合历史模型参数和当前模型参数，来平滑模型参数更新过程，从而提高模型的泛化能力。
    """
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)
""" 
如果代码运行到“p_src = param_dict_src[p_name]”这一行报错 KeyError，通常是由于源模型和目标模型的参数名称不一致导致的。

具体而言，param_dict_src是一个字典，它将源模型的参数名称映射为对应的参数对象。而在遍历目标模型的参数时，代码会尝试从param_dict_src中获取对应名称的源模型参数，如果找不到，则会报错 KeyError。

解决这个问题的方法是，检查源模型和目标模型的参数名称是否一致。如果不一致，可以通过修改代码来解决，或者手动将源模型的参数名称改为和目标模型一致。
"""

class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None, 
                 lorann=None, t2i_adapter=None,text_embedding = None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.lorann = lorann
        self.t2i_adapter = t2i_adapter
        self.text_embedding = text_embedding
    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))
                
    def save_lora(self,path):
        ## save lora weights 
        os.makedirs(path, exist_ok=True)
        lora_state={}
        # for name,param in self.nnet.named_parameters():
        #     name_cols=name.split('.')
        #     filter_names=['lora']
        #     if any(n==name_cols[-1] for n in filter_names):
        #        lora_state[name]=param
        #        print(name)
        for name,param in self.nnet.named_parameters():
            if 'lora' in name:
                lora_state[name]=param
        
        torch.save(lora_state,os.path.join(path,'lora.pt.tmp'))
        os.replace(os.path.join(path,'lora.pt.tmp'),os.path.join(path,'lora.pt'))

    def resume(self, ckpt_path=None, only_load_model=False):
        if ckpt_path is None:
            return

        logging.info(f'resume from {ckpt_path}, only_load_model={only_load_model}')
        self.step = torch.load(os.path.join(ckpt_path, 'step.pth'))

        if only_load_model:
            for key, val in self.__dict__.items():
                if key == 'nnet_ema' or key == 'nnet':
                    val.load_state_dict(torch.load(os.path.join(ckpt_path, f'{key}.pth'), map_location='cpu'))
        else:
            for key, val in self.__dict__.items():
                if key != 'step' and val is not None:
                    val.load_state_dict(torch.load(os.path.join(ckpt_path, f'{key}.pth'), map_location='cpu'))

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())

def initialize_train_state(config, device, uvit_class,text_encoder = None):
    
    nnet = uvit_class(**config.nnet)
    logging.info(f'load nnet from {config.nnet_path}')

    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'),False)
    nnet = get_peft_model(nnet,lora_config)
  
    # nnet.load_state_dict(torch.load('/home/wuyujia/competition/model_output/girl1_new_10000/lora.pt.tmp/lora.pt', map_location='cpu'), False)
    nnet.print_trainable_parameters()
    

    input_embed_params = list(text_encoder.get_input_embeddings().parameters())
    param_lists = input_embed_params + [param for name, param in nnet.named_parameters() if 'lora' in name]
    
    # for i in range(15):
    #     param_lists.append(nnet.in_blocks[i].attn.parameters())
    #     param_lists.append(nnet.out_blocks[i].attn.parameters())    
    # for i in range(15):
    #     param_lists.append(nnet.in_blocks[i].lora_attention.parameters())
    #     param_lists.append(nnet.out_blocks[i].lora_attention.parameters())
    # param_lists = [
    #     text_encoder.get_input_embeddings().parameters(),
    #     nnet.parameters()]

    nnet_ema = uvit_class(**config.nnet)
    nnet_ema.eval()
    # param_lists = list(itertools.chain(*param_lists))
    
    # logging.info(f'nnet has {cnt_params(nnet)} parameters')
    # logging.info(f'text_encoder has {cnt_params(text_encoder)} parameters')
  
    optimizer = get_optimizer(param_lists, **config.optimizer)
  
    lr_scheduler = get_lr_scheduler(optimizer,**config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema, text_embedding=text_encoder.get_input_embeddings())
    # train_state.ema_update(0)
    train_state.to(device)
    # no need to resume
    # train_state.resume(config.resume_ckpt_path, only_load_model=config.only_load_model)

    # for the case when the lr is manually changed
    lr_scheduler.base_lrs = [config.optimizer.lr]
    optimizer.param_groups[0]['initial_lr'] = config.optimizer.lr
    lr_scheduler._last_lr = lr_scheduler.get_lr()
    optimizer.param_groups[0]['lr'] = lr_scheduler.get_lr()[0]

    return train_state



def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.'):
            hparam_full, val = argv[i].split('=')
            hparam = hparam_full.split('.')[-1]
            if hparam_full.startswith('--config.optimizer.lm'):
                hparam = f'lm_{hparam}'
            if hparam_full.startswith('--config.optimizer.decoder'):
                hparam = f'decoder_{hparam}'
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams

def add_prefix(dct, prefix):
    return {f'{prefix}/{key}': val for key, val in dct.items()}

def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def setup(config):
    import builtins
    import ml_collections
    from torch import multiprocessing as mp
    import accelerate
    import wandb
    import logging
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration
    logging_dir = Path('./model_output/', './model_output/logs')
    accelerator_project_config = ProjectConfiguration(project_dir='./model_output/', logging_dir=logging_dir)
    mp.set_start_method('spawn')
    assert config.gradient_accumulation_steps == 1, \
        'fix the lr_scheduler bug before using larger gradient_accumulation_steps'
    
    logger = get_logger(__name__)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps,mixed_precision=None, project_config=accelerator_project_config)
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
         #初始化跟踪器，指定项目名称为 "unidiffuser"，同时传递参数配置 vars(args)
        accelerator.init_trackers("unidiffuser", config=vars(config))
    # if accelerator.is_main_process:
    #     wandb.init(dir=os.path.abspath(config.workdir), project='lora', config=config.to_dict(), job_type='train', mode="offline")
    #     set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
    #     logging.info(config)
    # else:
    #     set_logger(log_level='error')
    #     builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    return accelerator, device



def get_data_generator(loader, enable_tqdm, desc):
    while True:
        for data in tqdm(loader, disable=not enable_tqdm, desc=desc):
            yield data

