#!/bin/bash

# 定义需要执行的命令
declare -a arr=("CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/girl1_new_1000 --prompt_path eval_prompts/girl1_edit.json --output_path outputs/girl1_edit_1000"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/girl1_new_1500 --prompt_path eval_prompts/girl1_edit.json --output_path outputs/girl1_edit_1500"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/girl1_new_2000 --prompt_path eval_prompts/girl1_edit.json --output_path outputs/girl1_edit_2000"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/girl1_new_2500 --prompt_path eval_prompts/girl1_edit.json --output_path outputs/girl1_edit_2500"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/girl1_new_2500 --prompt_path eval_prompts/girl1_sim.json --output_path outputs/girl1_sim_2500"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/girl1_new_1500 --prompt_path eval_prompts/girl1_sim.json --output_path outputs/girl1_sim_1500"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/girl1_new_2000 --prompt_path eval_prompts/girl1_sim.json --output_path outputs/girl1_sim_2000"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/girl1_new_1000 --prompt_path eval_prompts/girl1_sim.json --output_path outputs/girl1_sim_1000")

# 循环检查GPU状态并执行命令
for i in "${arr[@]}"; do
  # 检查2号显卡（CUDA_VISIBLE_DEVICES=2）状态
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i 2)" != "" ]; do
    # 如果2号显卡不空闲，则等待一段时间再次检查
    echo "gpu is busy, waiting for it to become free..."
    sleep 20
  done

  # 执行命令
  echo "Executing command: $i"
  eval $i
done

echo "All commands have been executed. Exiting..."







