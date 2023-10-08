import torch
import argparse
from peft import inject_adapter_in_model, LoraConfig,get_peft_model
lora_config = LoraConfig(
   inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1,target_modules=["qkv","fc1","fc2","proj","text_embed","clip_img_embed"]
)

def compress(delta_ckpt, ckpt, compression_ratio=0.6, device='cuda'):
    st = torch.load(f'{delta_ckpt}')


    compressed_key = 'state_dict'
    compressed_st = {compressed_key: {}}
    
    pretrained_st = torch.load(f'{ckpt}')
    pretrained_st = pretrained_st.state_dict()
   
    if 'embed' in st:
        """如果原始模型中存在名为embed的权重，则将其添加到compressed_st字典中，并从原始模型的权重中删除embed项。"""
        compressed_st['embed'] = st['embed']
        del st['embed']
        
    

    print("getting compression")
    layers = list(st.keys()) # 获取原始模型权重中的所有层的名称，并将其存储在layers变量中。
    for name in layers:
        """对于每个层的名称，如果名称中包含to_k或to_v，则执行以下操作：
            将原始模型的权重W和预训练模型的权重Wpretrain转换为device上的张量。
            计算deltaW，即W和Wpretrain之间的差异。
            对deltaW执行奇异值分解（SVD）。
            计算解释方差比（explained variance ratio），直到其超过compression_ratio。
            将压缩后的权重添加到compressed_st字典中。
            如果名称中不包含to_k或to_v，则将原始模型的权重添加到compressed_st字典中。
        """
        if 'lora' in name:
            W = st[name].to(device)
            Wpretrain = pretrained_st[name].clone().to(device)
            deltaW = W-Wpretrain

            u, s, vt = torch.linalg.svd(deltaW.clone())

            explain = 0
            all_ = (s).sum()
            for i, t in enumerate(s):
                explain += t/(all_)
                if explain > compression_ratio:
                    break

            compressed_st[compressed_key][f'{name}'] = {}
            compressed_st[compressed_key][f'{name}']['u'] = (u[:, :i]@torch.diag(s)[:i, :i]).clone()
            compressed_st[compressed_key][f'{name}']['v'] = vt[:i].clone()
        else:
            compressed_st[compressed_key][f'{name}'] = st[name]

   

    torch.save(compressed_st, f'/home/wuyujia/competition/model_output/girl2_copy/lora.pt.tmp/compressed_lora.pt')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--delta_ckpt',default='/home/wuyujia/competition/model_output/girl2_copy/lora.pt.tmp/lora.pt', help='path of checkpoint to compress',
                        type=str)
    parser.add_argument('--ckpt',default='nnet.pt', help='path of pretrained model checkpoint',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compress(args.delta_ckpt, args.ckpt)
