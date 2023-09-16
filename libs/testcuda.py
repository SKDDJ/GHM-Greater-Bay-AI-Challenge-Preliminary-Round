# #code for checking the cuda edition
# import torch
# import cuda
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(cuda.__version__)

# 查看当前的 cuda 版本的代码
import torch
print(torch.version.cuda)