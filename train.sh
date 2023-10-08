# #!/bin/bash
# accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d train_data/boy1 -o model_output/boy1
# accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d train_data/boy2 -o model_output/boy2
# accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d train_data/girl1 -o model_output/girl1
# accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d train_data/girl2 -o model_output/girl2


#!/bin/bash
accelerate launch --mixed_precision no --num_processes 1 train.py --instance_data_dir="train_data/newboy1"  --outdir="model_output/boy1"  --class_data_dir="real_reg/samples_boyface"   --with_prior_preservation  --prior_loss_weight=1.0  --class_prompt="boy" --num_class_images=200  --instance_prompt=" a <new1> boy"  --modifier_token "<new1>"
accelerate launch --mixed_precision no --num_processes 1 trainn.py --instance_data_dir="train_data/newboy2"  --outdir="model_output/boy2"  --class_data_dir="real_reg/samples_boyface"   --with_prior_preservation  --prior_loss_weight=1.0  --class_prompt="boy" --num_class_images=200  --instance_prompt=" a <new1> boy"  --modifier_token "<new1>"
accelerate launch --mixed_precision no --num_processes 1 trainn.py --instance_data_dir="train_data/newgirl1" --outdir="model_output/girl1" --class_data_dir="real_reg/samples_girlhead"  --with_prior_preservation  --prior_loss_weight=1.0  --class_prompt="girl" --num_class_images=200 --instance_prompt=" a <new1> girl" --modifier_token "<new1>"
accelerate launch --mixed_precision no --num_processes 1 trainn.py --instance_data_dir="train_data/newgirl2" --outdir="model_output/girl2" --class_data_dir="real_reg/samples_girlhead"  --with_prior_preservation  --prior_loss_weight=1.0  --class_prompt="girl" --num_class_images=200 --instance_prompt=" a <new1> girl" --modifier_token "<new1>"

python --mixed_precision no --num_processes 1 trainn.py --instance_data_dir="train_data/newgirl2" --outdir="model_output/girl2" --class_data_dir="real_reg/samples_girlhead"  --with_prior_preservation  --prior_loss_weight=1.0  --class_prompt="girl" --num_class_images=200 --instance_prompt=" a <new1> girl" --modifier_token "<new1>"