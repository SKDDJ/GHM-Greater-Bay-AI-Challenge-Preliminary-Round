#!/bin/bash
accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d train_data/boy1 -o model_output/boy1
accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d train_data/boy2 -o model_output/boy2
accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d train_data/girl1 -o model_output/girl1
accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d train_data/girl2 -o model_output/girl2