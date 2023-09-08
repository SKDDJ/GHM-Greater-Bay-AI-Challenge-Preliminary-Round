import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.pretrained_path = "/home/shiyiming/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/b95be7d6f134c3a9e62ee616f310733567f069ce"
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1
    config.gradient_accumulation_steps = 1

    config.log_interval = 50
    config.eval_interval = 100
    
    
    config.save_interval = 200

    
    config.max_step = 200
    config.batch_size = 2
    
    config.center_crop = True
    config.real_prior = True
    config.reversion = None
    
    config.nnet_path = "models/uvit_v1.pth"
    
    config.max_grad_norm = 1.0
    
    config.dataloader_num_workers = 10 # original is 10
    
   #  config.sample_batch_size = 4  it seems uesless...
    config.revision = None
    config.num_workers = 10
    # config.batch_size = 6
    
    
    config.resolution = 512
    
    config.clip_img_model = "ViT-B/32"
    # config.clip_text_model = "openai/clip-vit-large-patch14"
    config.clip_text_model = "/home/schengwei/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff"
    
    config.only_load_model = True
    

    config.optimizer = d(
        name='adamw',
        lr=2e-05, # for custom diffusion, lr=5e-6, but in code is will double if with preservation is True.
        weight_decay=0.03, # 0.03
        betas=(0.9, 0.999),# adam_beta2 changed from 0.9 to 0.999, same as Custom Diffusion.
        amsgrad=False,
        eps=1e-8, # adam_eps added 
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=500
    )

    config.autoencoder = d(
        pretrained_path='models/autoencoder_kl.pth',
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.get_ref('text_dim'),
        tokenizer_path = "./models/gpt2"
    )

    config.nnet = d(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.get_ref('text_dim'),
        num_text_tokens=77,
        clip_img_dim=config.get_ref('clip_img_dim'),
        use_checkpoint=True
    )


    # sample

    config.mode = "t2i"
    config.n_samples = 9 # control the numbers of generating images 
    config.n_iter = 1 # 过多的迭代次数可能导致过拟合或生成的样本过于接近训练数据
    config.nrow = 4
    config.sample = d(
        sample_steps=100, # 我从 30 调到了 100
        scale=7.,
        t2i_cfg_mode='true_uncond' # 笑死，之前sample 中的也是用的是true_uncond模式，生成的图片能看才见鬼
    )

    return config
