"""
采样代码
文件输入:
    prompt, 指定的输入文件夹路径, 制定的输出文件夹路径
文件输出:
    采样的图片, 存放于指定输出文件夹路径
- 指定prompt文件夹位置, 选手需要自行指定模型的地址以及其他微调参数的加载方式, 进行图片的生成并保存到指定地址, 此部分代码选手可以修改。
- 输入文件夹的内容组织方式和训练代码的输出一致
- sample的方法可以修改
- 生成过程中prompt可以修改, 但是指标测评时会按照给定的prompt进行测评。
"""
import os
import ml_collections
import torch
import random
import argparse
import utils
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image, make_grid
import numpy as np
import clip
import time
from libs.clip import FrozenCLIPEmbedder
import numpy as np
import json
from libs.uvit_multi_post_ln_v1 import UViT
from peft import inject_adapter_in_model, LoraConfig,get_peft_model
lora_config = LoraConfig(
   inference_mode=False, r=128, lora_alpha=90, lora_dropout=0.1,target_modules=["qkv","fc1","fc2","proj","to_out","to_q","to_k","to_v","text_embed","clip_img_embed"]
)


def get_model_size(model):
    """
    统计模型参数大小
    """
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    return para

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(config, clip_text_model, autoencoder):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    contexts = torch.randn(config.n_samples, 77, config.clip_text_dim).to(device)
    img_contexts = torch.randn(config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    clip_imgs = torch.randn(config.n_samples, 1, config.clip_img_dim)

    prompts = [ config.prompt ] * config.n_samples
    contexts = clip_text_model.encode(prompts)

    return contexts, img_contexts, clip_imgs


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample(prompt_index, config, nnet, clip_text_model, autoencoder, device):
    """
    using_prompt: if use prompt as file name
    """
    n_iter = config.n_iter
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    config = ml_collections.FrozenConfigDict(config)

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)


    use_caption_decoder = config.text_dim < config.clip_text_dim or config.mode != 't2i'
    if use_caption_decoder:
        from libs.caption_decoder import CaptionDecoder
        caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    else:
        caption_decoder = None

    empty_context = clip_text_model.encode([''])[0]

    def split(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        return z, clip_img


    def combine(z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)


    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = split(x)

        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out = combine(z_out, clip_img_out)

        if config.sample.scale == 0.:
            return x_out

        if config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            if use_caption_decoder:
                _empty_context = caption_decoder.encode_prefix(_empty_context)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + config.sample.scale * (x_out - x_out_uncond)



    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)


    contexts, img_contexts, clip_imgs = prepare_contexts(config, clip_text_model, autoencoder)
    contexts_low_dim = contexts if not use_caption_decoder else caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet

    _n_samples = contexts_low_dim.size(0)


    def sample_fn(**kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
        _x_init = combine(_z_init, _clip_img_init)

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return t2i_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)

        _z, _clip_img = split(x)
        return _z, _clip_img


    samples = None    
    for i in range(n_iter):
        _z, _clip_img = sample_fn(text=contexts_low_dim)  # conditioned on the text embedding
        new_samples = unpreprocess(decode(_z))
        if samples is None:
            samples = new_samples
        else:
            samples = torch.vstack((samples, new_samples))

    os.makedirs(config.output_path, exist_ok=True)
    for idx, sample in enumerate(samples):
        save_path = os.path.join(config.output_path, f'{prompt_index}-{idx:03}.jpg')
        save_image(sample, save_path)
        
    print(f'results are saved in {save_path}')


def compare_model(standard_model:torch.nn.Module, model:torch.nn.Module, mapping_dict= {}):
    """
    compare model parameters based on paramter name
    for parameters with same name(common key), directly compare the paramter conetent
    all other parameters will be regarded as differ paramters, except keys in mapping_dict.
    mapping_dict is a python dict class, with keys as a subset of `origin_only_keys` and values as a subset of `compare_only_keys`.
    
    """
    origin_dict = dict(standard_model.named_parameters())
    origin_keys = set(origin_dict.keys())
    compare_dict = dict(model.named_parameters())
    compare_keys = set(compare_dict.keys())
    
    origin_only_keys = origin_keys - compare_keys
    compare_only_keys = compare_keys - origin_keys
    common_keys = set.intersection(origin_keys, compare_keys)
    
    
    diff_paramters = 0
    # compare parameters of common keys
    for k in common_keys:
        origin_p = origin_dict[k]
        compare_p = compare_dict[k]
        if origin_p.shape != compare_p.shape:
            diff_paramters += origin_p.numel() + compare_p.numel()
        elif (origin_p - compare_p).norm() != 0:
            diff_paramters += origin_p.numel()
    
    
    mapping_keys = set(mapping_dict.keys())
    assert set.issubset(mapping_keys, origin_only_keys)
    assert set.issubset(set(mapping_dict.values()), compare_only_keys)
    
    for k in mapping_keys:
        origin_p = origin_dict[k]
        compare_p = compare_dict[mapping_keys[k]]
        if origin_p.shape != compare_p.shape:
            diff_paramters += origin_p.numel() + compare_p.numel()
        elif (origin_p - compare_p).norm() != 0:
            diff_paramters += origin_p.numel()
    # all keys left are counted
    extra_origin_keys = origin_only_keys - mapping_keys
    extra_compare_keys = compare_only_keys - set(mapping_dict.values())
    
    for k in extra_compare_keys:
        diff_paramters += compare_dict[k]
    
    for k in extra_origin_keys:
        diff_paramters += origin_dict[k]    
    
    return diff_paramters

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_path", type=str, default="models/uvit_v1.pth", help="nnet path to resume")
    parser.add_argument("--prompt_path", type=str, default="eval_prompts/boy1.json", help="file contains prompts")
    parser.add_argument("--output_path", type=str, default="outputs/boy1", help="output dir for generated images")
    parser.add_argument("--weight_dir", type=str, default="model_output/girl2", help="output dir for weights of text encoder")
    return parser.parse_args()


def main(argv=None):
    # config args
    from configs.sample_config import get_config
    set_seed(42)
    config = get_config()
    args = get_args()
    config.output_path = args.output_path
    # config.nnet_path = os.path.join(args.restore_path, "final.ckpt",'nnet.pth')
    config.lora_path = os.path.join(args.restore_path, "lora.pt.tmp",'lora.pt')
    config.n_samples = 3
    config.n_iter = 1
    device = "cuda"

    # init models
    nnet = UViT(**config.nnet)
    print(f'load nnet from {config.lora_path}')
    
    nnet.load_state_dict(torch.load("models/uvit_v1.pth", map_location='cpu'), False)

    nnet = get_peft_model(nnet,lora_config)
    # nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'),True)
    nnet.load_state_dict(torch.load(config.lora_path, map_location='cpu'), False)
    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    clip_text_model.load_textual_inversion(args.weight_dir, token = "<new1>" , weight_name="<new1>.bin")
    clip_text_model.to("cpu")
    
    
    nnet_mapping_dict = {}
    autoencoder_mapping_dict = {}
    clip_text_mapping_dict = {}
    
    total_diff_parameters = 0
    
    nnet_standard = UViT(**config.nnet)
    # nnet_standard = get_peft_model(nnet_standard,lora_config)
    nnet_standard.load_state_dict(torch.load("models/uvit_v1.pth", map_location='cpu'), False)
    nnet_standard = get_peft_model(nnet_standard,lora_config)
    total_diff_parameters += compare_model(nnet_standard, nnet, nnet_mapping_dict)
    del nnet_standard
    
    autoencoder_standard = libs.autoencoder.get_model(**config.autoencoder)
    total_diff_parameters += compare_model(autoencoder_standard, autoencoder, autoencoder_mapping_dict)
    del autoencoder_standard
    
    clip_text_strandard = FrozenCLIPEmbedder(version=config.clip_text_model, device=device).to("cpu")
    total_diff_parameters += compare_model(clip_text_strandard, clip_text_model, clip_text_mapping_dict)
    del clip_text_strandard
    
    clip_text_model.to(device)
    autoencoder.to(device)
    nnet.to(device)
    
    # 基于给定的prompt进行生成
    prompts = json.load(open(args.prompt_path, "r"))
    for prompt_index, prompt in enumerate(prompts):
        # 根据训练策略
        if "boy" in prompt:
            prompt = prompt.replace("boy", "<new1> boy")
        else:
            prompt = prompt.replace("girl", "<new1> girl")

        config.prompt = prompt
        print("sampling with prompt:", prompt)
        sample(prompt_index, config, nnet, clip_text_model, autoencoder, device)
        
    print(f"\033[91m finetuned parameters: {total_diff_parameters}\033[00m")

if __name__ == "__main__":
    main()