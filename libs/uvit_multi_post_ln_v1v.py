import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, DropPath, Mlp
import einops
import torch.utils.checkpoint
import torch.nn.functional as F


class LoraLayer(nn.Module):
    def __init__(self,raw_linear,in_features,out_features,r,alpha):
        super().__init__()
        self.r=r 
        self.alpha=alpha
        self.lora_a=nn.Parameter(torch.empty((in_features,r)))
        self.lora_b=nn.Parameter(torch.zeros((r,out_features)))
    
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))

        self.raw_linear=raw_linear
    
    def forward(self,x):    # x:(batch_size,in_features)
        raw_output=self.raw_linear(x)   
        lora_output=x@((self.lora_a@self.lora_b)*self.alpha/self.r)    # matmul(x,matmul(lora_a,lora_b)*alpha/r)
        return raw_output+lora_output

def inject_lora(model,name,layer):
    name_cols=name.split('.')

    # 逐层下探到linear归属的module
    children=name_cols[:-1]
    cur_layer=model 
    for child in children:
        cur_layer=getattr(cur_layer,child)
    
    #print(layer==getattr(cur_layer,name_cols[-1]))
    lora_layer=LoraLayer(layer,layer.in_features,layer.out_features,24,1)
    setattr(cur_layer,name_cols[-1],lora_layer)



class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=24, network_alpha=None, device='cuda:0', dtype=None):
        super().__init__()
        
        
        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank
        
        return up_hidden_states.to(orig_dtype)



if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'uvit attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, in_chans):
    patch_size = int((x.shape[2] // in_chans) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * in_chans == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


def interpolate_pos_emb(pos_emb, old_shape, new_shape):
    pos_emb = einops.rearrange(pos_emb, 'B (H W) C -> B C H W', H=old_shape[0], W=old_shape[1])
    pos_emb = F.interpolate(pos_emb, new_shape, mode='bilinear')
    pos_emb = einops.rearrange(pos_emb, 'B C H W -> B (H W) C')
    return pos_emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            with torch.amp.autocast(device_type='cuda', enabled=False):
                qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
                q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LoraAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,rank = 24, network_alpha = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.to_q_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        q = self.to_q_lora(x)
        k = self.to_k_lora(x)
        v = self.to_v_lora(x)
        # qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            q = einops.rearrange(q, 'B L (H D) -> B L H D', H=self.num_heads)
            k = einops.rearrange(k, 'B L (H D) -> B L H D', H=self.num_heads)
            v = einops.rearrange(v, 'B L (H D) -> B L H D', H=self.num_heads)
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            with torch.amp.autocast(device_type='cuda', enabled=False):
                qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
                q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.to_out_lora(x)
        x = self.proj_drop(x)
        return x
    


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim) if skip else None
        self.norm2 = norm_layer(dim)
        #self.lora_attention = LoraAttention(dim, num_heads=num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint
    def add_lora(self,Lora):
        self.lora_attn = Lora
        
    def forward(self, x, skip=None, lora_input_img = None, lora_input_text = None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip, lora_input_img, lora_input_text)
        else:
            return self._forward(x, skip, lora_input_img, lora_input_text)

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
            x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))
        x = self.norm2(x)

        x = x + self.drop_path(self.mlp(x))
        x = self.norm3(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class UViT(nn.Module):
    def __init__(self, img_size, in_chans, patch_size, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, pos_drop_rate=0., drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, mlp_time_embed=False, use_checkpoint=False,
                 text_dim=None, num_text_tokens=None, clip_img_dim=None):
        super().__init__()

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

      
        
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size  # the default img size
        assert self.img_size[0] % patch_size == 0 and self.img_size[1] % patch_size == 0
        self.num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)

        self.time_img_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.time_text_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.text_embed = nn.Linear(text_dim, embed_dim)
        self.text_out = nn.Linear(embed_dim, text_dim)

        self.clip_img_embed = nn.Linear(clip_img_dim, embed_dim)
        self.clip_img_out = nn.Linear(embed_dim, clip_img_dim)

        self.num_text_tokens = num_text_tokens
        self.num_tokens = 1 + 1 + num_text_tokens + 1 + self.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, skip=True, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        
        # print("lora_attention",self.in_blocks[0].lora_attention.to_q_lora.up.weight)
        # exit()
        self.token_embedding = nn.Embedding(2, embed_dim)
        self.pos_embed_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    def add_lora(self,Lora):
        self.Lora = Lora

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, img, clip_img, text, t_img, t_text, data_type):
        _, _, H, W = img.shape

        img = self.patch_embed(img)

        t_img_token = self.time_img_embed(timestep_embedding(t_img, self.embed_dim))
        t_img_token = t_img_token.unsqueeze(dim=1)
        t_text_token = self.time_text_embed(timestep_embedding(t_text, self.embed_dim))
        t_text_token = t_text_token.unsqueeze(dim=1)

        text = self.text_embed(text)
        clip_img = self.clip_img_embed(clip_img)
        token_embed = self.token_embedding(data_type).unsqueeze(dim=1)
        
        
        #     t_img_token torch.Size([2, 1, 1536])
        #     t_text_token torch.Size([2, 1, 1536])
        #     token_embed torch.Size([2, 1, 1536])
        #     text torch.Size([2, 77, 1536])
        #     clip_img torch.Size([2, 1, 1536])
        #     img torch.Size([2, 1024, 1536])
            
        x = torch.cat((t_img_token, t_text_token, token_embed, text, clip_img, img), dim=1)
        num_text_tokens, num_img_tokens = text.size(1), img.size(1)

        pos_embed = torch.cat(
            [self.pos_embed[:, :1 + 1, :], self.pos_embed_token, self.pos_embed[:, 1 + 1:, :]], dim=1)
        if H == self.img_size[0] and W == self.img_size[1]:
            pass
        else:  # interpolate the positional embedding when the input image is not of the default shape
            pos_embed_others, pos_embed_patches = torch.split(pos_embed, [1 + 1 + 1 + num_text_tokens + 1, self.num_patches], dim=1)
            pos_embed_patches = interpolate_pos_emb(pos_embed_patches, (self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size),
                                                    (H // self.patch_size, W // self.patch_size))
            pos_embed = torch.cat((pos_embed_others, pos_embed_patches), dim=1)

        x = x + pos_embed
        x = self.pos_drop(x)
        skips = []
        count = 0
        for blk in self.in_blocks:
            x = blk(x)
            count += 1
            skips.append(x)
   
        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skip = skips.pop())

        x = self.norm(x)

        t_img_token_out, t_text_token_out, token_embed_out, text_out, clip_img_out, img_out = x.split((1, 1, 1, num_text_tokens, 1, num_img_tokens), dim=1)
        

        img_out = self.decoder_pred(img_out)
        img_out = unpatchify(img_out, self.in_chans)

        clip_img_out = self.clip_img_out(clip_img_out)

        text_out = self.text_out(text_out)
        return img_out, clip_img_out, text_out



    # lora权重的加载
    try:
        restore_lora_state=torch.load('lora.pt')
        nnet.load_state_dict(restore_lora_state,strict=False)
    except:
        raise ValueError("lora.pt not found")