import os
from PIL import Image

def resize_images_in_path(image_path):
    # 设置目标尺寸
    target_sizes = {
        "girl1": (1080, 1440),
        "girl2": (3024, 4032),
        "boy1": (480, 640),
        "boy2": (1280, 1710),
    }

    # 确定目标尺寸
    target_size = None
    for key in target_sizes.keys():
        if key in image_path:
            target_size = target_sizes[key]
            break

    if target_size is None:
        print("No matching keyword found in path.")
        return

    # 遍历image_path目录下的所有文件
    for filename in os.listdir(image_path):
        # 如果文件是.jpg文件
        if filename.endswith(".jpg"):
            # 打开图像
            image = Image.open(os.path.join(image_path, filename))
            # 调整图像尺寸
            image = image.resize(target_size, resample=Image.BICUBIC)
            # 保存结果到原来的位置，覆盖原来的文件
            image.save(os.path.join(image_path, filename))
