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
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


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


def initialize_train_state(config, device, uvit_class):
    clip_text_model = CLIPEmbedder(version=config.clip_text_model, device=device)
    params = []
    nnet = uvit_class(**config.nnet)
    params = list(itertools.chain(clip_text_model.transformer.get_input_embeddings().parameters(), nnet.lora_adapters_itot.parameters(), nnet.lora_adapters_ttoi.parameters()))
    nnet_ema = uvit_class(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema, text_embedding=clip_text_model.transformer.get_input_embeddings())
    train_state.ema_update(0)
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
         #初始化跟踪器，指定项目名称为 "custom-diffusion"，同时传递参数配置 vars(args)
        accelerator.init_trackers("custom-diffusion", config=vars(config))
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

