# 代码说明

## 预测性能

预计训练和推理总耗时10小时左右

## 环境配置（必选）

在 Dockerfile 中有每一步的注释，详情见 Dockerfile 文件

## 数据（必选）

* 使用了CelebA（CelebFaces Attribute）人脸属性数据集，由香港中文大学开放提供，数据获取链接为http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html，随机获得200张男性人脸（在训练boy1和boy2模型时作为正则数据集使用）和200张女性人脸

  * 在训练girl1和girl2模型时作为正则数据集使用
* 对官方提供的数据集进行移除背景和裁剪人像的操作，放进train_data_crop文件夹内

  * 在 train.sh文件中指定训练数据集的时候使用

## 预训练模型（必选）

* 使用了 CompVis/stable-diffusion-v1-4中的 AutoTokenizer， 代码继承自 custom diffusion中微调 CLIP Text Encoder 部分
  * 路径位于other_models/stablediffusion/ 下
  * 并且代码中应该会自动从 hugging face 上拉取

## 算法（必选）

* 整体思路介绍（必选）

[Texual Inversion]中微调CLIP Text Encoder 的方法 -> 为防止过拟合采用[DreamBooth]中的Prior Loss方法 -> 为 UViT 的指定 Layer 加入 LoRA

* 方法的创新点

之前在训练步数较少的情况下采用过 cross attention 的 LoRA

* 算法的其他细节

1. 自定义Token和CLIP Tokenizer更新：通过添加一个近义词的自定义token，扩展了CLIP模型的文本处理能力。更新CLIP Tokenizer，以包括新的token，确保模型能够识别和处理它。更新CLIPTextModel的embedding层，以适应新的token；
2. 人脸正则数据集：为了避免模型的过拟合和语言漂移问题，引入了人脸正则数据集，帮助模型学习更好地理解人脸。训练数据集由 instance data 和 class data 组成，计算loss时，将模型输出分开，分别算loss；
3. 参数微调：利用peft库中的lora，对模型的"qkv","fc1","fc2","proj","text_embed","clip_img_embed"部分进行参数微调。

### 整体思路介绍（必选）

1. 用近义词初始化 `<new1>`，微调 CLIP Text Encoder
2. 为了避免过拟合和语言漂移，加入人脸正则数据集；
3. 利用peft库中的lora，对模型进行参数微调。

训练代码说明:

基于单个数据集进行训练:

```shell
accelerate launch --mixed_precision fp16 --num_processes 1 train.py -d '<训练文件所在位置>' -o '<模型输出>' 
```

训练所有数据集:

```shell
./train.sh
```

训练过程中的数据预处理, 模型架构, 加载方式等等均可进行修改, 只需要满足命令行接口即可, 并且模型输出位置的输出形式能够被 `sample.py`文件正确解析即可。

生成代码说明:

基于prompt文件生成图片:

```shell
python sample.py --restore_path '<模型输出>' --prompt_path '<prompt文件路径>' --output_path '<输出路径>'
```

基于所有的prompt文件进行生成:

```
./sample.sh
```

文件中sample的方式, prompt均可以更改, 但是测评时只会根据文件中的prompt进行测评。每个prompt要求输出三张图片, 并存放于目标路径。模型输出路径只需要能够正确解析训练产生的文件夹即可(若使用高效参数微调方法, 只需要将额外参数保存到输出路径并在sample.py中加载即可, 无需保存整个模型, 原模型可以从 `models/uvit_v1.pth`中加载)。`sample.py`除了会生成图片, 还会检查用于生成的模型和原模型的参数差异大小, 用于衡量微淘的参数量, 具体的计算方式见代码。

### 客观指标打分(部分)

```shell
python score.py

# 默认路径为
    # --dataset: './train_data/'
    # --prompts: './eval_prompts/'
    # --outputs: './outputs/'

# 可自行提供路径
python score.py --dataset '<数据路径>' --prompts '<prompt路径>' --outputs '<提交路径>'
```

## 训练流程（必选）

1. 定义命令行参数，包括数据目录和输出目录。
2. 在loop()函数中，使用train_state对象进行模型训练。
3. 在训练过程中，使用accelerator.is_main_process判断当前进程是否为主进程。
4. 如果是主进程，计算当前步数total_step，并在达到一定步数时记录日志和保存模型。
5. 在达到一定步数时，保存模型的checkpoint文件，以便后续进行模型推理。
6. 在训练结束时，保存最终的模型checkpoint文件。

## 其他注意事项

* 运行代码命令须参考修改过的train.sh和sample.sh文件
* 控制训练步数（以“图文对”为单位）的参数所在位置：workspace路径下train.py 文件第396行
* 因为我们初赛最终版本train_step为 10000步，并且使用了multi_stage的训练方法，即训练中途更换过训练集图片，这一版复现分数应该达不到初赛最终版那么高。
