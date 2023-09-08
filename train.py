"""
训练代码
代码输入:
    - 数据文件夹路径, 其中包含近近脸照文件夹和全身照文件夹, 
    - 指定的输出路径, 用于输出模型
    - 其他的参数需要选手自行设定
代码输出:
    - 微调后的模型以及其他附加的子模块
    
accelerate launch train.py \
  --instance_data_dir ="目标图像的数据集路径" \
  --outdir="自己的模型输出路径"\
  --class_data_dir "自己的正则数据集路径" \
  --with_prior_preservation  --prior_loss_weight=1.0 \
  --class_prompt="girl" --num_class_images=200 \
  --instance_prompt="photo of a <new1> girl"  \
  --modifier_token "<new1>"
"""
from accelerate import Accelerator
import hashlib
import warnings
import torch
import utils
from absl import logging
import os
#import wandb
import libs.autoencoder
import clip
import itertools
from libs.clip import CLIPEmbedder
from libs.caption_decoder import CaptionDecoder
from torch.utils.data import DataLoader
from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I
import argparse
import yaml
import datetime
from transformers import AutoTokenizer,PretrainedConfig
from pathlib import Path
from libs.data import PersonalizedBase, PromptDataset, collate_fn
from libs.uvit_multi_post_ln_v1 import UViT
# import diffusers
# from diffusers import DiffusionPipeline
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from pathlib import Path
from transformers import CLIPTextModel
import tqdm

from accelerate.logging import get_logger 
import itertools
import json
#from pathos.multiprocessing import ProcessingPool as Pool
from peft import inject_adapter_in_model, LoraConfig,get_peft_model
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=24,
    bias="none",
    target_modules=["qkv","proj"],
)


# 保存text encoder中新增token的embedding

def save_new_embed(clip_text_model, modifier_token_id, accelerator, args, outdir):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(clip_text_model).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        torch.save(learned_embeds_dict, f"{outdir}/{y}.bin")

logger = get_logger(__name__)

def freeze_params(params):
    for param in params:
        param.requires_grad = False
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """
    根据预训练模型的名称或路径导入相应的模型类。

    参数：
    pretrained_model_name_or_path: 预训练模型的名称或路径。
    revision: 模型的版本号。

    返回：
    模型类。

    根据模型配置获取模型类，支持的模型包括 CLIPTextModel 和 RobertaSeriesModelWithTransformation。
    如果模型类不在支持列表中，将引发 ValueError 异常。
    """
    # 从预训练配置中获取文本编码器配置
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    # 获取模型类名
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel

        return RobertaSeriesModelWithTransformation
    else:
        # 模型类不在支持列表中，引发 ValueError 异常
        raise ValueError(f"{model_class} is not supported.")
     
        
        


def train(config):
    
    """
    prepare models
    准备各类需要的模型
    """
    accelerator, device = utils.setup(config)

    args = get_args()
    concepts_list = args.concepts_list
    # concepts_list = [
    #         {
    #             "instance_prompt": 'photo of a <new1> girl', #photo of a <new1> girl
    #             "class_prompt": 'girl',#girl
    #             "instance_data_dir": './train_data/oldgirl2',#./train_data/girl2
    #             "class_data_dir": './real_reg/samples_girlbody/',#./real_reg/samples_person/
    #         }
    #     ]    
       # Generate class images if prior preservation is enabled.

    if config.with_prior_preservation:
        for i, concept in enumerate(concepts_list):
            # 目录文件处理
            class_images_dir = Path(concept["class_data_dir"])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            if config.real_prior:
                assert (
                    class_images_dir / "images"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {config.num_class_images}"
                assert (
                    len(list((class_images_dir / "images").iterdir())) == config.num_class_images
                ), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {config.num_class_images}"
                assert (
                    class_images_dir / "caption.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {config.num_class_images}"
                assert (
                    class_images_dir / "images.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {config.num_class_images}"
                concept["class_prompt"] = os.path.join(class_images_dir, "caption.txt")
                concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
                concepts_list[i] = concept
                accelerator.wait_for_everyone()
            
    pretrained_model_name_or_path = "/home/wuyujia/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision = None,
            use_fast=False,
            )
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path , config.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision
    )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision
    # )
    text_encoder.to(device)
    train_state = utils.initialize_train_state(config, device, uvit_class=UViT,text_encoder = text_encoder)
    logging.info(f'load nnet from {config.nnet_path}')
    
    train_state.nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'), False)

 
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)


    nnet, optimizer = accelerator.prepare(train_state.nnet, train_state.optimizer)
    nnet.to(device)
    # nnet = get_peft_model(nnet,lora_config)
    # for i in range (15):
    #         module = nnet.in_blocks[i].attn
    #         module = inject_adapter_in_model(lora_config, module)
    # module = nnet.mid_block
    # module = inject_adapter_in_model(lora_config, module)
    # for i in range (15):
    #         module = nnet.out_blocks[i].attn
    #         module = inject_adapter_in_model(lora_config, module)
    # print("success_add_lora")       
    # 全参微调不加lora
    # for name,param in nnet.named_parameters():
    #     param.requires_grad=True
    # for name,param in nnet.named_parameters():
    #      if 'lora_adapters_ttoi' in name or 'lora_adapters_itot'  in name:
    #         param.requires_grad = False  
    
    
    # 非Lora部分不计算梯度
    for name,param in nnet.named_parameters():
        if 'lora_attention' in name or 'token_embedding' in name:
            param.requires_grad = True
        else:
            param.requires_grad=False

    # for name,param in nnet.named_parameters():
    #     if 'lora' in name or 'token_embedding' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad=False

    # for name,param in nnet.named_parameters():
    #     if 'lora_attention' in name or 'token_embedding' in name or 'lora_adapters_ttoi' in name or 'lora_adapters_itot' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad=False
            
    # check the nnet's parameters if they are frozen
    # for name, param in nnet.named_parameters():
    #     print(f'{name}: requires_grad={param.requires_grad}') 
    
    lr_scheduler = train_state.lr_scheduler

    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    
    autoencoder.requires_grad = False
    


    
    # Modify the code of custom diffusion to directly import the clip text encoder 
    # instead of freezing all parameters.
    # clip_text_model = CLIPEmbedder(version=config.clip_text_model, device=device)


    clip_img_model, clip_img_model_preprocess = clip.load(config.clip_img_model, jit=False)
    # clip_img_model.to(device).eval().requires_grad_(False)
    clip_img_model.to(device).requires_grad_(False)
    
    # Adding a modifier token which is optimized #### 来自Textual inversion代码
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    # add modifier token
    modifier_token_id = []
    initializer_token_id = []

    if args.modifier_token is not None:
        
        args.modifier_token = args.modifier_token.split("+")#['<new1>']
        args.initializer_token = config.initializer_token.split("+")#['ktn', 'pll', 'ucd']

        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            #在添加占位符标记时，通常会将占位符添加到词汇表（vocabulary）中，
            #以便在处理文本时能够正确地处理这个占位符。占位符可以在模型训练、文本生成、填充序列等任务中起到重要的作用。
            
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            
            #[42170]
            #ktn
            
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")
            
            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))
            print("modifier_token_id",modifier_token_id)
        
        
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))#从40408变为40409

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)


    """
    处理数据部分
    """
    # process data
    train_dataset = PersonalizedBase(
                                     concepts_list=concepts_list,
                                     num_class_images=config.num_class_images,
                                     size=config.resolution, # 设置的默认为 512
                                     center_crop=config.center_crop,
                                     tokenizer_max_length=77,
                                     tokenizer=tokenizer,
                                     config = config,
                                     hflip=config.hflip,
                                    #  mask_size= autoencoder.encode(torch.randn(1, 3, config.resolution, config.resolution).to(dtype=torch.float16).to(accelerator.device)
                                    # )
                                    # .latent_dist.sample()
                                    # .size()[-1],
                                     mask_size= 64 #custom_diffusion里mask_size的值为64
                                    )
    train_dataset_loader = DataLoader(train_dataset,
                                      batch_size=2,
                                      shuffle=True,
                                      collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
                                      num_workers=0,
                                      )

    train_data_generator = utils.get_data_generator(train_dataset_loader, enable_tqdm=accelerator.is_main_process, desc='train')

    logging.info("saving meta data")
    os.makedirs(config.meta_dir, exist_ok=True)
    with open(os.path.join(config.meta_dir, "config.yaml"), "w") as f:
        f.write(yaml.dump(config))
        f.close()
    
    _betas = stable_diffusion_beta_schedule()
    schedule = Schedule(_betas)
    logging.info(f'use {schedule}')
    # for name, param in nnet.named_parameters():
    #         param.requires_grad = True
    # for name, param in nnet.named_parameters():
    #     if 'lora_adapters_itot' not in name and 'lora_adapters_ttoi' not in name:
    #         param.requires_grad = False
    # for name, param in nnet.named_parameters():
    #     if 'text_embed' in name or 'token_embedding' in name:
    #         param.requires_grad = True
    
    # 验证哪些参数被冻结
    for name, param in nnet.named_parameters():
        if  param.requires_grad:
            print(f"未冻结的参数: {name}")

    # total_frozen_params = sum(p.numel() for p in text_encoder.parameters() if  p.requires_grad)
 
    # 77560320 lora_adapter+text_embedding  37946112 token_embedding
    # INFO - nnet has 1029970000 parameters
    # INFO - text_encoder has 123060480 parameters
    # text_encoder = accelerator.prepare(text_encoder)
    def train_step():
        metrics = dict()
        
        text, img, img4clip, mask = next(train_data_generator)
        img = img.to(device)
        text = text.to(device)
        img4clip = img4clip.to(device)
        data_type = torch.float32
        mask = mask.to(device)
        # with torch.no_grad():
        z = autoencoder.encode(img)
        clip_img = clip_img_model.encode_image(img4clip).unsqueeze(1).contiguous()
        text = text_encoder(text)[0]
        text = caption_decoder.encode_prefix(text)
        #z= false text = true
       
        bloss = LSimple_T2I(img=z,clip_img=clip_img, text=text, data_type=data_type, nnet=nnet, schedule=schedule, device=device, config=config,mask=mask)
        # bloss.requires_grad = True
        
        accelerator.backward(bloss)
        for name, param in nnet.named_parameters():
            if param.grad is not None or 0:
                print(name)

        # for name, param in text_encoder.named_parameters():
        #     if param.grad is not None:
        #         print(name)
        # 如果参数的梯度不为None，说明存在梯度
        
       
        # Zero out the gradients for all token embeddings except the newly added
        # embeddings for the concept, as we only want to optimize the concept embeddings
        if True:
            # 谁给删了，而且改回来了，下面这个 if 语句没什么大用，都是一样的效果
            # if accelerator.num_processes > 1:
            #     grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
            # else:
            #     grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
            grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
            # Get the index for tokens that we want to zero the grads for
            index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
            for i in range(len(modifier_token_id[1:])):
                index_grads_to_zero = index_grads_to_zero & (
                    torch.arange(len(tokenizer)) != modifier_token_id[i]
                )
            grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                index_grads_to_zero, :
            ].fill_(0)


        
        params_to_clip = (
            itertools.chain(text_encoder.parameters(), nnet.parameters())
            if args.modifier_token is not None
            else nnet.parameters()
        )
        accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
        
        
        #  更新参数
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        
        optimizer.zero_grad()
        metrics['bloss'] = accelerator.gather(bloss.detach().mean()).mean().item()
        # metrics['loss_img'] = accelerator.gather(loss_img.detach().mean()).mean().item()
        # metrics['loss_clip_img'] = accelerator.gather(loss_clip_img.detach().mean()).mean().item()
        # metrics['scale'] = accelerator.scaler.get_scale()
        metrics['lr'] = train_state.optimizer.param_groups[0]['lr']
       
        return metrics

    # @torch.no_grad()
    # @torch.autocast(device_type='cuda')
    # def eval(total_step):
    #     """
    #     write evaluation code here
    #     """

    #     return

    def loop():
        log_step = config.log_interval 
        # log_step = 0
        # eval_step = 1000000
        save_step = config.save_interval # 100
        # save_step = 0
        count = 0
        while True:
            nnet.train()
            with accelerator.accumulate(nnet),accelerator.accumulate(text_encoder):
                metrics = train_step()
            print("metrics",metrics)
            count+=1
            print(count)
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                # nnet.eval()
                total_step = train_state.step * config.batch_size
                if total_step >= log_step:
                    logging.info(utils.dct2str(dict(step=total_step, **metrics)))
               #     wandb.log(utils.add_prefix(metrics, 'train'), step=total_step)
                    # train_state.save(os.path.join(config.log_dir, f'{total_step:04}.ckpt'))
                    log_step += config.log_interval

                # if total_step >= eval_step:
                #     eval(total_step)
                #     eval_step += config.eval_interval

                # if total_step >= config.save_interval :#save_step = 300
                #     logging.info(f'Save and eval checkpoint {total_step}...')
                #     train_state.save(os.path.join(config.ckpt_root, f'{total_step:04}.ckpt'))
                #     save_step += config.save_interval
                   
                if total_step >= 300:
                    logging.info(f"saving final ckpts to {config.outdir}...")
                    save_new_embed(text_encoder, modifier_token_id, accelerator, args, args.outdir)
                    train_state.save(os.path.join(config.outdir, 'final.ckpt'))
                    # train_state.save_lora(os.path.join(config.outdir, 'lora.pt.tmp'))
                    break


    loop()

def get_args():
    parser = argparse.ArgumentParser()
    # key args
    # parser.add_argument('-d', '--data', type=str, default="train_data/girl2", help="datadir")
    parser.add_argument('-o', "--outdir", type=str, default="model_ouput/girl2", help="output of model")
    # args of logging
    parser.add_argument("--logdir", type=str, default="logs", help="the dir to put logs")
    parser.add_argument("--nnet_path", type=str, default="models/uvit_v1.pth", help="nnet path to resume")
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    ) 
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " concepts_list, additional images will be sampled with class_prompt."
        ),
    )

    # parser.add_argument(
    #     "--logging_dir",
    #     type=str,
    #     default="logs",
    #     help=(
    #         "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
    #         " *outdir/runs/**CURRENT_DATETIME_HOSTNAME***."
    #     ),
    # )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--real_prior",
        default=True,
        action="store_true",
        help="real images as prior.",
    )

    parser.add_argument("--modifier_token", type=str, default="<new1>", help="modifier token")
    parser.add_argument(
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    
    

    args = parser.parse_args()
    
    if args.with_prior_preservation:
        if args.concepts_list is None:
            args.concepts_list = [
                {
                    "instance_prompt": args.instance_prompt, #photo of a <new1> girl
                    "class_prompt": args.class_prompt,#girl
                    "instance_data_dir": args.instance_data_dir,#./path-to-images/
                    "class_data_dir": args.class_data_dir,#./real_reg/samples_person/
                }
            ]
          
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.concepts_list is not None:
            warnings.warn("You need not use --concepts_list without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")



    return args

def main():
    print("main start!")
    # 赛手需要根据自己的需求修改config file
    from configs.unidiffuserv1 import get_config
    config = get_config()
    config_name = "unidiffuserv1"
    args = get_args()
    config.log_dir = args.logdir
    config.outdir = args.outdir
    config.data = args.instance_data_dir
    config.modifier_token = args.modifier_token
    config.initializer_token = args.initializer_token
    config.prior_loss_weight = args.prior_loss_weight
    config.instance_prompt = args.instance_prompt
    config.class_prompt = args.class_prompt
    
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.with_prior_preservation = args.with_prior_preservation
    
    config.real_prior = args.real_prior
    config.num_class_images = args.num_class_images
    config.hflip = args.hflip
    
    data_name = Path(config.data).stem

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    config.workdir = os.path.join(config.log_dir, f"{config_name}-{data_name}-{now}")
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.meta_dir = os.path.join(config.workdir, "meta")
    config.nnet_path = args.nnet_path
    os.makedirs(config.workdir, exist_ok=True)

    train(config)




if __name__ == "__main__":
    main()


""" 
accelerate launch train.py \
  --instance_data_dir="/home/wuyujia/competition/train_data/newboy1" \
  --outdir="/home/wuyujia/competition/model_output/boy11"\
  --class_data_dir="/home/wuyujia/competition/real_reg/samples_boyface" \
  --with_prior_preservation  --prior_loss_weight=1.0 \
  --class_prompt="boy" --num_class_images=200 \
  --instance_prompt=" a <new1> boy"  \
  --modifier_token "<new1>"
"""