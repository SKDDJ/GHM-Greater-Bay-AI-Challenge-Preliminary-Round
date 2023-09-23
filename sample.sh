#!/bin/bash
# python sample.py --restore_path model_output/boy11 --prompt_path eval_prompts/boy1.json --output_path outputs/boy1 --weight_dir model_output/boy1
# python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts/boy2.json --output_path outputs/boy2 --weight_dir model_output/boy2
# python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts/girl1.json --output_path outputs/girl1 --weight_dir model_output/girl1
# python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts/girl2.json --output_path outputs/girl2 --weight_dir model_output/girl2

# python sample.py --restore_path model_output/boy11 --prompt_path eval_prompts/boy1.json --output_path outputs/boy1 --weight_dir model_output/boy1
# python sample.py --restore_path model_output/boy2withcrossattention --prompt_path eval_prompts/boy2.json --output_path outputs/boy2 --weight_dir model_output/boy2withcrossattention
# python sample.py --restore_path model_output/girl1withcrossattention --prompt_path eval_prompts/girl1.json --output_path outputs/girl1 --weight_dir model_output/girl1withcrossattention
# python sample.py --restore_path model_output/girl2withcrossattention --prompt_path eval_prompts/girl2.json --output_path outputs/girl2 --weight_dir model_output/girl2withcrossattention


python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_sim.json --output_path outputs/boy1_sim --weight_dir model_output/boy1
python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts_advance/boy2_sim.json --output_path outputs/boy2_sim --weight_dir model_output/boy2
python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_sim.json --output_path outputs/girl1_sim  --weight_dir model_output/girl1
python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_sim.json --output_path outputs/girl2_sim  --weight_dir model_output/girl2
python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_edit.json --output_path outputs/boy1_edit   --weight_dir model_output/boy1
python sample.py --restore_path model_output/boy2  --prompt_path eval_prompts_advance/boy2_edit.json --output_path outputs/boy2_edit  --weight_dir model_output/boy2
python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_edit.json --output_path outputs/girl1_edit  --weight_dir model_output/girl1
python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_edit.json --output_path outputs/girl2_edit    --weight_dir model_output/girl2

python sample.py --restore_path model_output/boy1 --prompt_path /home/wuyujia/competition/eval_prompts/boy1.json --output_path outputs/boy1 --weight_dir model_output/boy1
/home/wuyujia/competition/eval_prompts/boy1.json