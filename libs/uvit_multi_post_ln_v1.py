import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, DropPath, Mlp
import einops
import torch.utils.checkpoint
import torch.nn.functional as F

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=128, network_alpha=84, device='cuda:0', dtype=None):
        super().__init__()
        
        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        
        ### 当时在这里修改的 lora 初始化是 0 还是 1 等
        
        # nn.init.ones_(self.up.weight)
        nn.init.zeros_(self.up.weight)
        # self.up.weight.data.fill_(0.0001)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
    
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank
        
        return up_hidden_states.to(orig_dtype)

class lora_cross_attention_ttoi(nn.Module):
    def __init__(self, img_dim=1024, rank=24, text_dim=77, heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,proj_drop=0., hidden_size=1536, dropout = 0.0, network_alpha=None):

        super().__init__()
        self.heads = heads
        self.scale = qk_scale or (hidden_size//self.heads) ** -0.5
        self.hidden_size = hidden_size
        self.img_dim = img_dim
        self.text_dim = text_dim
        self.rank = rank
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.zeros_(self.to_q.weight)
        nn.init.zeros_(self.to_k.weight)
        nn.init.zeros_(self.to_v.weight)
        nn.init.zeros_(self.to_out.weight)
    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor
    
    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    def prepare_attention_mask(self, attention_mask, target_length, batch_size=2, out_dim=3):

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask
    
    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

    
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs
    
    def forward(self, img , text, attention_mask=None):
        batch_size = 2
        #添加mask
        attention_mask = self.prepare_attention_mask(attention_mask, 1024, batch_size)
        
        query = self.to_q(img)#得到query架构仍然为1024*1536
        key = self.to_k(text)
        value =self.to_v(text)#仍为77*1536
        # key = key.to(attn.to_q.weight.dtype)
        # value = value.to(attn.to_q.weight.dtype)

        #优化方法，可选择添加
        detach = torch.ones_like(key)
        detach[:, :1, :] = detach[:, :1, :] * 0.0
        key = detach * key + (1 - detach) * key.detach()
        value = detach * value + (1 - detach) * value.detach()


        #多头注意力
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        attention_probs = self.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
    
        # linear proj
        hidden_states = self.to_out(hidden_states)
        
        # dropout
        hidden_states = self.proj_drop(hidden_states)
        
        return hidden_states
    
class lora_cross_attention_itot(nn.Module):
    def __init__(self, img_dim=1024, rank=24, text_dim=77, heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,proj_drop=0., hidden_size=1536, dropout = 0.0, network_alpha=None):

        super().__init__()
        self.heads = heads
        self.scale = qk_scale or (hidden_size//self.heads) ** -0.5
        self.hidden_size = hidden_size
        self.img_dim = img_dim
        self.text_dim = text_dim
        self.rank = rank
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.zeros_(self.to_q.weight)
        nn.init.zeros_(self.to_k.weight)
        nn.init.zeros_(self.to_v.weight)
        nn.init.zeros_(self.to_out.weight)
    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor
    
    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    def prepare_attention_mask(self, attention_mask, target_length, batch_size=2, out_dim=3):

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask
    
    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

    
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs
    
    def forward(self, img , text, attention_mask=None):
        batch_size = 2
        #添加mask
        attention_mask = self.prepare_attention_mask(attention_mask, 1024, batch_size)
        
        query = self.to_q(text)#得到query架构仍然为77*1536
        key = self.to_k(img)
        value =self.to_v(img)#仍为77*1536
        # key = key.to(attn.to_q.weight.dtype)
        # value = value.to(attn.to_q.weight.dtype)

        #优化方法，可选择添加
        detach = torch.ones_like(key)
        detach[:, :1, :] = detach[:, :1, :] * 0.0
        key = detach * key + (1 - detach) * key.detach()
        value = detach * value + (1 - detach) * value.detach()


        #多头注意力
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        attention_probs = self.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
    
        # linear proj
        hidden_states = self.to_out(hidden_states)
        
        # dropout
        hidden_states = self.proj_drop(hidden_states)
 
        return hidden_states




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
        self.to_qkv_lora = LoRALinearLayer(dim , dim*3 ,rank ,network_alpha)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.to_qkv_lora(x)
        
        
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            
            # 原始注释写的写法和这个差不多捏，只是不是单独进行处理了捏
            
            # q = einops.rearrange(q, 'B L (H D) -> B L H D', H=self.num_heads)
            # k = einops.rearrange(k, 'B L (H D) -> B L H D', H=self.num_heads)
            # v = einops.rearrange(v, 'B L (H D) -> B L H D', H=self.num_heads)
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

        x = self.to_out(x)
        x = self.proj_drop(x)
        return x
    


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim) if skip else None
        self.norm2 = norm_layer(dim)
        # self.lora_attention = LoraAttention(dim, num_heads=num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
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

    def _forward(self, x, skip, lora_input_img, lora_input_text):
        if  hasattr (self,'lora_attn'):
            if self.skip_linear is not None:
                x = self.skip_linear(torch.cat([x, skip], dim=-1))
                x = self.norm1(x)
            if lora_input_text is not None and lora_input_img is not None: 
                x_attention = self.lora_attention(x) 
                x = x + self.drop_path(self.attn(x)) + x_attention 
                t_img_token, t_text_token, token_embed, text, clip_img, img = x.split((1, 1, 1, 77, 1, 1024), dim=1)
                text= text + lora_input_text
                img= img + lora_input_img
                x = torch.cat((t_img_token, t_text_token, token_embed, text, clip_img, img), dim=1)
                x = self.norm2(x)
            else:
                x_attention = self.lora_attention(x)
                x = x + self.drop_path(self.attn(x))+ self.drop_path(x_attention) 
                x = self.norm2(x)

            x = x + self.drop_path(self.mlp(x))
            x = self.norm3(x)
        else:
            if self.skip_linear is not None:
                x = self.skip_linear(torch.cat([x, skip], dim=-1))
                x = self.norm1(x)
            if lora_input_text is not None and lora_input_img is not None:
                x = x + self.drop_path(self.attn(x))
                t_img_token, t_text_token, token_embed, text, clip_img, img = x.split((1, 1, 1, 77, 1, 1024), dim=1)
                text= text + lora_input_text
                img= img + lora_input_img
                x = torch.cat((t_img_token, t_text_token, token_embed, text, clip_img, img), dim=1)
                x = self.norm2(x)
            else:
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
        
        ## lora 的初始化
            
        self.adapters_itot = nn.ModuleList()
        for _ in range(30):
            self.adapters_itot.append(lora_cross_attention_itot())
        
        self.adapters_ttoi = nn.ModuleList()
        for _ in range(30):
            self.adapters_ttoi.append(lora_cross_attention_ttoi())
        
        # print("lora_attention",self.in_blocks[0].lora_attention.to_q.up.weight)
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
            if not hasattr(self, 'Lora'):
                t_img_token, t_text_token, token_embed, text, clip_img, img = x.split((1, 1, 1, num_text_tokens, 1, num_img_tokens), dim=1)                
                modelttoi = self.adapters_ttoi[count]  
                modelitot = self.adapters_itot[count]
                modelttoi.to('cuda')
                modelitot.to('cuda')
            
                lora_img = modelttoi(img,text)  
                lora_text = modelitot(img,text)
                x = torch.cat((t_img_token, t_text_token, token_embed, text, clip_img, img), dim=1)            
                x = blk(x, skip = None, lora_input_img = lora_img,lora_input_text = lora_text)    
                count += 1           
            else:
                x = blk(x)
                count += 1
            skips.append(x)
   
        x = self.mid_block(x)

        for blk in self.out_blocks:
            ## 虽然这里我感觉多此一举的添加了一个变量 y，还没有看懂用意，但是时候 maybe 需要花时间看看
            
            if not hasattr(self, 'Lora'):
                skip = skips.pop()
                y = x
                y = blk.skip_linear(torch.cat([y, skip], dim=-1))
                y = blk.norm1(y)
                t_img_token, t_text_token, token_embed, text, clip_img, img = y.split((1, 1, 1, num_text_tokens, 1, num_img_tokens), dim=1)                
                modelttoi = self.adapters_ttoi[count]  
                modelitot = self.adapters_itot[count]
                modelttoi.to('cuda')
                modelitot.to('cuda')
                lora_img = modelttoi(img,text)  
                lora_text = modelitot(img,text)
                del y         
                x = blk(x, skip, lora_input_img = lora_img,lora_input_text = lora_text)    
                count += 1
            else:
                x = blk(x, skip = skips.pop(), lora_input_img = None, lora_input_text = None)
                count += 1

        x = self.norm(x)

        t_img_token_out, t_text_token_out, token_embed_out, text_out, clip_img_out, img_out = x.split((1, 1, 1, num_text_tokens, 1, num_img_tokens), dim=1)
        

        img_out = self.decoder_pred(img_out)
        img_out = unpatchify(img_out, self.in_chans)

        clip_img_out = self.clip_img_out(clip_img_out)

        text_out = self.text_out(text_out)
        return img_out, clip_img_out, text_out


