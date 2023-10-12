#!/bin/bash

# 定义需要执行的命令
declare -a arr=("CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_3000 --prompt_path eval_prompts/boy1_edit.json --output_path outputs/boy1_edit_3000"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_3500 --prompt_path eval_prompts/boy1_edit.json --output_path outputs/boy1_edit_3500"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_4000 --prompt_path eval_prompts/boy1_edit.json --output_path outputs/boy1_edit_4000"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_2000 --prompt_path eval_prompts/boy1_edit.json --output_path outputs/boy1_edit_2000"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_2500 --prompt_path eval_prompts/boy1_edit.json --output_path outputs/boy1_edit_2500"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_2500 --prompt_path eval_prompts/boy1_sim.json --output_path outputs/boy1_sim_2500"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_3500 --prompt_path eval_prompts/boy1_sim.json --output_path outputs/boy1_sim_3500"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_2000 --prompt_path eval_prompts/boy1_sim.json --output_path outputs/boy1_sim_2000"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_3000 --prompt_path eval_prompts/boy1_sim.json --output_path outputs/boy1_sim_3000"
                "CUDA_VISIBLE_DEVICES=2 python sample.py --restore_path model_output/boy1_new_4000 --prompt_path eval_prompts/boy1_sim.json --output_path outputs/boy1_sim_4000"
                "CUDA_VISIBLE_DEVICES=2 python score.py"
               )

# 循环检查GPU状态并执行命令
for i in "${arr[@]}"; do
  # 检查0号显卡状态
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i 2)" != "" ]; do
    # 如果0号显卡不空闲，则等待一段时间再次检查
    echo "gpu is busy, waiting for it to become free..."
    sleep 20
  done

  # 执行命令
  echo "Executing command: $i"
  eval $i
done

echo "All commands have been executed. Exiting..."
