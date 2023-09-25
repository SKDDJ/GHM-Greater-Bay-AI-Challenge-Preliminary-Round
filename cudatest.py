import os
import torch
os.environ['CUDA_VISIBLE_DEVICES']="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.randn((200000, 300, 200, 20), device=device)
