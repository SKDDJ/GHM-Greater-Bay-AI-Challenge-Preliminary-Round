import torch

# 在适当的位置调用以释放 GPU 内存
torch.cuda.empty_cache()