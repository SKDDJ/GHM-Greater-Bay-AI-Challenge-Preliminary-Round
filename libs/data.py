from PIL import Image
import io
import torchvision.transforms as transforms
import numpy as np
import re
import os
import PIL
from PIL import Image
from torch.utils.data import Dataset
import random
from PIL.ImageOps import exif_transpose
import torch
from pathlib import Path



def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]#实例id
    pixel_values = [example["instance_images"] for example in examples]#实例图像
    clip_img = [example["instance_clip_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        clip_img += [example["class_clip_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    mask = torch.cat(mask , dim = 0)
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "clip_img": clip_img,
        "mask": mask,
    }
    clip_img = torch.cat(clip_img, dim=0)
    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask
    
    return input_ids,pixel_values,clip_img,mask


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class PersonalizedBase(Dataset):
    def __init__(self,
                 concepts_list,
                 size=512,
                 mask_size=64,
                 center_crop=False,
                 tokenizer_max_length=None,
                 num_class_images=200,
                 tokenizer=None,
                 config = None,
                 hflip=False,
                 aug=True,
                 ):
        self.size = size
        self.mask_size = mask_size
        
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        
        self.tokenizer_max_length = tokenizer_max_length
        
        self.interpolation = PIL.Image.BILINEAR
        self.aug = aug
        
        self.instance_images_path = []
        self.class_images_path = []
        
        for concept in concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            class_data_root = Path(concept["class_data_dir"])
        
           
            if os.path.isdir(class_data_root):
                class_images_path = list(class_data_root.iterdir())
                print("class_images_path",class_images_path)
                class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
            else:
                with open(class_data_root, "r") as f:
                    class_images_path = f.read().splitlines()
                with open(concept["class_prompt"], "r") as f:
                    class_prompt = f.read().splitlines()

            class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
            self.class_images_path.extend(class_img_path[:num_class_images])
        self.transform_clip = _transform(224)
        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length
    
    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_clip_images"] = self.transform_clip(instance_image)
        instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt
        
        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        
        class_image, class_prompt = self.class_images_path[index % self.num_class_images]
        print(class_image,class_prompt)
        class_image = Image.open(class_image)
        if not class_image.mode == "RGB":
            class_image = class_image.convert("RGB")
        example["class_images"] = self.image_transforms(class_image)
        example["class_mask"] = torch.ones_like(example["mask"])
        example["class_prompt_ids"] = self.tokenizer(
            class_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["class_clip_images"] = self.transform_clip(class_image)
        
        return example
