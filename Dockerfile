FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /workspace

RUN apt update && apt upgrade -y
RUN apt install -y git libgl1-mesa-glx libglib2.0-0
RUN apt-get install -y gcc g++

## dependency installation
COPY ./CLIP ./CLIP
COPY ./requirements.txt ./requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install -r ./requirements.txt

COPY . .

