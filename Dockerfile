# 使用基础镜像 pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 设置工作目录为 /workspace
WORKDIR /workspace

## 依赖安装
# 复制项目目录中的 CLIP 文件夹到容器的 /workspace/CLIP 目录下
COPY ./CLIP ./CLIP

# 复制项目目录中的 requirements.txt 文件到容器的 /workspace/requirements.txt 目录下
COPY ./requirements.txt ./requirements.txt

# 更新系统包列表并升级已安装的包
RUN apt update && apt upgrade -y

# 安装所需的系统依赖包
RUN apt install -y git libgl1-mesa-glx libglib2.0-0

# 安装gcc和g++编译器
RUN apt-get install -y gcc g++

# 设置Python的全局镜像源为清华大学的镜像，加速包的下载
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 

# 使用pip安装项目所需的Python依赖包，包括requirements.txt中列出的依赖
RUN pip install -r ./requirements.txt

# 复制项目目录中的所有文件到容器的当前工作目录
COPY . .
