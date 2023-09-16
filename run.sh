#!/bin/bash
tar_file=$1
device=$2

sha256=`docker load --input $tar_file | grep -Po "sha256:(\w+)" | sed 's/sha256:\(.*\)/\1/g'`

docker run -it --gpus "device=${device}" --rm -v /home/test01/eval_prompts_advance:/workspace/eval_prompts_advance -v /home/test01/train_data:/workspace/train_data -v /home/test01/models:/workspace/models \
-v /root/indocker_shell.sh:/workspace/indocker_shell.sh $sha256