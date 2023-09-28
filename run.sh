#!/bin/bash
tar_file=$1
device=$2

sha256=`docker load --input $tar_file | grep -Po "sha256:(\w+)" | sed 's/sha256:\(.*\)/\1/g'`


docker run -it --gpus "device=${device}" --rm -v /home/test01/eval_prompts_advance:/workspace/eval_prompts_advance -v /home/test01/train_data:/workspace/train_data -v /home/test01/models:/workspace/models \
-v /root/indocker_shell.sh:/workspace/indocker_shell.sh $sha256



# docker run -it --gpus "device=${device}" --rm -v /home/wuyujia/competition/eval_prompts_advance:/workspace/eval_prompts_advance 
# -v /home/wuyujia/competition/train_data:/workspace/train_data -v /home/wuyujia/competition/models:/workspace/models \
# -v /home/wuyujia/competition/indocker_shell.sh:/workspace/indocker_shell.sh $sha256 

sudo docker run -it --gpus all --rm -v /home/wuyujia/competition/eval_prompts_advance:/workspace/eval_prompts_advance -v /home/wuyujia/competition/train_data:/workspace/train_data -v /home/wuyujia/competition/models:/workspace/models -v /home/wuyujia/competition/indocker_shell.sh:/workspace/indocker_shell.sh  -v /home/wuyujia/competition/sample.py:/workspace/sample.py -v /home/wuyujia/.insightface:/root/.insightface -v /home/wuyujia/.cache/huggingface:/root/.cache/huggingface xiugou:v1


