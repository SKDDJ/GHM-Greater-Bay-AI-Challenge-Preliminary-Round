"""
推理速度和内存使用标准程序
给定模型,随机种子,采样步数以及采样方法以获取固定数量的图片
在一定的差异容忍度下测量生成图片需要的时间以及显存占用情况
注意: 程序速度和显存优化并非赛题的主要部分, 分数权重待定, 请赛手做好各个子任务之间的平衡
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
from PIL import Image
import clip
import time
from libs.clip import FrozenCLIPEmbedder
import numpy as np
from libs.uvit_multi_post_ln_v1 import UViT
from libs.caption_decoder import CaptionDecoder

from peft import inject_adapter_in_model, LoraConfig,get_peft_model
lora_config = LoraConfig(
   r=128, lora_alpha=90, lora_dropout=0.1,target_modules=["qkv","fc1","fc2","proj","to_out","to_q","to_k","to_v","text_embed","clip_img_embed"]
#    target_modules=["qkv","fc1","fc2","proj"]
)



def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(config, clip_text_model, clip_img_model, clip_img_model_preprocess, autoencoder):
    """
    准备上下文数据集。

    参数：
    config: 配置信息。
    clip_text_model: 文本编码模型，用于将文本转化为嵌入向量。
    clip_img_model: 图像编码模型。
    clip_img_model_preprocess: 图像预处理模型。
    autoencoder: 自编码器模型。

    返回：
    contexts: 文本上下文数据，形状为 (n_samples, 77, config.clip_text_dim)。
    img_contexts: 图像上下文数据，形状为 (n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])。
    clip_imgs: 图像数据，形状为 (n_samples, 1, config.clip_img_dim)。

    准备上下文数据集的过程：
    1. 创建指定形状的随机上下文数据。
    2. 使用指定的 prompt 对上下文数据进行编码，得到文本上下文数据。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 创建随机的上下文数据
    contexts = torch.randn(config.n_samples, 77, config.clip_text_dim).to(device)
    img_contexts = torch.randn(config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    clip_imgs = torch.randn(config.n_samples, 1, config.clip_img_dim)
    # 使用指定的 prompt 对上下文数据进行编码，得到文本上下文数据
    prompts = [ config.prompt ] * config.n_samples
    contexts = clip_text_model.encode(prompts)
    # 返回准备好的上下文数据
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


def sample(config, nnet, clip_text_model, autoencoder, clip_img_model, 
           clip_img_model_preprocess, caption_decoder, device):
    """
    使用指定配置对模型进行采样。
    
    参数：
    config: 配置信息，采用 `ml_collections` 库的 `FrozenConfigDict` 类。
    nnet: 神经网络模型。
    clip_text_model: 文本编码模型，用于将文本转化为嵌入向量。
    autoencoder: 自编码器模型，用于特征编码和解码。
    clip_img_model: 图像编码模型，用于将图像转化为嵌入向量。
    clip_img_model_preprocess: 图像预处理模型，用于图像处理前的预处理操作。
    caption_decoder: 标题解码器，用于将嵌入向量转化为标题文本。


    使用指定的配置参数对模型进行采样，并返回结果。
    """
    n_iter = config.n_iter
    use_caption_decoder = True
    if config.get('benchmark', False):
         # 若配置中存在 benchmark 设置，则设置 cudnn 的相关参数以进行性能优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    config = ml_collections.FrozenConfigDict(config)


    ############ start timing #############
    start_time = time.time()
    
    #训练时的超参数
    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    # 空白上下文，用于将空字符串编码为文本嵌入向量
    empty_context = clip_text_model.encode([''])[0]

    def split(x):
        """
        将输入张量 x 拆分为 z 和 clip_img 两部分。

        参数：
        x: 输入张量，包含 z 和 clip_img 两个部分。

        返回：
        z: 经过重新排列后的 z 部分张量，形状为 (B, C, H, W)。
        clip_img: 经过重新排列后的 clip_img 部分张量，形状为 (B, L, D)，其中 L 为 1。
        """
        # 从配置中获取 z_shape 的各维度大小
        C, H, W = config.z_shape
        # 计算 z 部分的维度
        z_dim = C * H * W
        # 使用 split 函数将 x 按照指定维度拆分为 z 和 clip_img
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
        # 对 z 部分进行 einops 重新排列，将 'B (C H W)' 转换为 'B C H W' 形状
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        # 对 clip_img 部分进行 einops 重新排列，将 'B (L D)' 转换为 'B L D' 形状
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        return z, clip_img


    def combine(z, clip_img):
        """
        将 z 和 clip_img 部分重新组合为一个张量。

        参数：
        z: 经过重新排列的 z 部分张量，形状为 (B, C, H, W)。
        clip_img: 经过重新排列的 clip_img 部分张量，形状为 (B, L, D)，其中 L 为 1。

        返回：
        组合后的张量，形状为 (B, C*H*W + L*D)。
        """
         # 对 z 部分进行 einops 重新排列，将 'B C H W' 转换为 'B (C H W)' 形状
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
         # 对 clip_img 部分进行 einops 重新排列，将 'B L D' 转换为 'B (L D)' 形状
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        # 使用 torch.cat 函数将 z 和 clip_img 张量沿着最后一个维度连接起来
        return torch.concat([z, clip_img], dim=-1)


    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        """
        对输入 x 进行 T2I 网络的操作，生成融合了条件和无条件输出的结果。

        参数：
        x: 输入张量，用于 T2I 网络的操作。
        timesteps: 时间步信息张量。
        text: 文本嵌入向量的低维版本。

        返回：
        融合了条件和无条件输出的张量。

        算法步骤：
        1. 计算条件模型输出。
        2. 计算无条件模型输出。
        - 若 config.sample.t2i_cfg_mode 为 'empty_token'，使用带有空字符串的原始配置。
        - 若 config.sample.t2i_cfg_mode 为 'true_uncond'，使用通过我们方法学习的无条件模型。
        3. 返回条件输出和无条件输出的线性组合。
        """
        # 将输入张量 x 拆分为 z 和 clip_img 部分
        z, clip_img = split(x)
        # 创建时间步信息张量
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        # 使用 nnet 进行条件模型的操作
        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        # 将 z_out 和 clip_img_out 部分重新组合为 x_out
        x_out = combine(z_out, clip_img_out)
        # 若配置中的 scale 为 0，则直接返回 x_out
        if config.sample.scale == 0.:
            return x_out
        # 根据配置中的 t2i_cfg_mode 执行相应的无条件模型操作
        if config.sample.t2i_cfg_mode == 'empty_token':
            # 使用重复空上下文的方式得到无条件模型的输出
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            if use_caption_decoder:
                _empty_context = caption_decoder.encode_prefix(_empty_context)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            # 将 z_out_uncond 和 clip_img_out_uncond 部分重新组合为 x_out_uncond
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            # 使用真实的无条件模型学习到的文本进行操作
            text_N = torch.randn_like(text)  # 3 other possible choices
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
             # 将 z_out_uncond 和 clip_img_out_uncond 部分重新组合为 x_out_uncond
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            # 若配置中的 t2i_cfg_mode 不属于上述两种模式，则抛出未实现错误
            raise NotImplementedError
        # 返回融合了条件和无条件输出的张量
        return x_out + config.sample.scale * (x_out - x_out_uncond)


    @torch.cuda.amp.autocast()
    def decode(_batch):
        """
        对批量数据进行解码
        """
        return autoencoder.decode(_batch)

    # 准备上下文数据集
    contexts, img_contexts, clip_imgs = prepare_contexts(config, clip_text_model, clip_img_model, clip_img_model_preprocess, autoencoder)
    # 若使用标题解码器，则将上下文数据编码为低维版本，作为 nnet 模型的输入
    # 这个低维版本的上下文数据是 nnet 模型的输入
    contexts_low_dim = contexts if not use_caption_decoder else caption_decoder.encode_prefix(contexts)  
    # 获取低维版本上下文数据的样本数量
    _n_samples = contexts_low_dim.size(0)


    def sample_fn(**kwargs):
        """
        样本生成函数，使用 DPM_Solver 采样生成样本。

        参数：
        **kwargs: 其他关键字参数。

        返回：
        _z: 生成的 z 部分样本。
        _clip_img: 生成的 clip_img 部分样本。

        生成过程：
        1. 创建初始的 _z_init 和 _clip_img_init。
        2. 创建噪声时间表。
        3. 定义 model_fn 函数，用于计算 t2i_nnet 模型的输出。
        4. 使用 DPM_Solver 进行采样，生成样本 x。
        5. 将生成的 x 拆分为 _z 和 _clip_img 部分样本。

        注意：生成样本时使用了自动混合精度（autocast）。
        """
        # 创建随机初始的 _z_init 和 _clip_img_init
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
        _x_init = combine(_z_init, _clip_img_init)
        # 创建噪声时间表
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())
        # 定义 model_fn 函数，用于计算 t2i_nnet 模型的输出
        def model_fn(x, t_continuous):
            t = t_continuous * N
            return t2i_nnet(x, t, **kwargs)
        # 使用 DPM_Solver 进行采样，生成样本 x
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):

            x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
            
        # 将生成的 x 拆分为 _z 和 _clip_img 部分样本
        _z, _clip_img = split(x)
        return _z, _clip_img

    
    # 迭代 n_iter 次，生成样本
    samples = None  
   
    for i in range(n_iter):
        # 调用 sample_fn 函数生成样本，条件为低维版本的上下文数据
        _z, _clip_img = sample_fn(text=contexts_low_dim)  
        
        # 对生成的 _z 进行解码和反预处理，得到新的样本
        new_samples = unpreprocess(decode(_z))
        # 将新生成的样本添加到 samples 中
        if samples is None:
            samples = new_samples
        else:
            samples = torch.vstack((samples, new_samples))

    ############# end timing ##############
    end_time = time.time()

    # 文件需要保存为jpg格式
    os.makedirs(config.output_path, exist_ok=True)
    
    # 遍历生成的样本并保存为 jpg 格式文件
    for idx, sample in enumerate(samples):
        # 构造保存路径，以 prompt 和索引号命名
        save_path = os.path.join(config.output_path, f'{config.prompt}-{idx:03}.jpg')
        # 使用 save_image 函数保存样本为 jpg 文件
        save_image(sample, save_path)
        

    print(f'\nresults are saved in {os.path.join(config.output_path)} :)')
    mem_use = torch.cuda.max_memory_reserved()
    print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
    print(f"\nusing time: {end_time - start_time:.2f}s")
    return (mem_use, end_time - start_time)



def assert_same(path1, path2):
    img1_list = os.listdir(path1)
    img2_list = os.listdir(path2)

    if len(img1_list) != len(img2_list):
         return False
        
    def eval_single(img1, img2):
        img1 = np.array(Image.open(img1))
        img2 = np.array(Image.open(img2))
        mean_diff = np.linalg.norm(img1 - img2)/(512*512)
        if mean_diff < 1:
            return True
        else:
            return False

    for img1, img2 in zip(img1_list, img2_list):
        if eval_single(os.path.join(path1, img1), os.path.join(path2, img2)):
            continue
        else:
            return False
    return True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnet_path", type=str, default="models/uvit_v1.pth", help="nnet path to resume")
    parser.add_argument("--output_path", type=str, default="bench_samples", help="path to place output imgs")
    parser.add_argument("--half", action="store_true", help="half precision for memory optiomization")
    return parser.parse_args()


def main(argv=None):
    # config args
    from configs.sample_config import get_config
    set_seed(42)
    config = get_config()
    args = get_args()
    config.nnet_path = args.nnet_path
    config.output_path = args.output_path

    config.n_samples = 2
    config.n_iter = 15
    device = "cuda"

    # init models
    nnet = UViT(**config.nnet)
    
    Lora = True
    print(config.nnet_path)
    print(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'),False)
    nnet = get_peft_model(nnet,lora_config)
    nnet.to(device)
    if args.half:
        nnet = nnet.half()
    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    clip_img_model, clip_img_model_preprocess = clip.load(config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    config.prompt = "a white girl with green hair"

    sample(config, nnet, clip_text_model, autoencoder, clip_img_model, 
           clip_img_model_preprocess, caption_decoder, device)
    
    # if assert_same("bench_samples_standard", config.output_path):
    #     print("error assertion passed")
    # else:
    #     print("error assertion failed")
    

if __name__ == "__main__":
    main()