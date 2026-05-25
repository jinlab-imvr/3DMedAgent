import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# Determine the data directory dynamically
# amlt_data_dir = os.getenv("AMLT_DATA_DIR", "/mnt/default/data")
# path_to_check = "/mnt/external/data"
# DATA_DIR = path_to_check if os.path.exists(path_to_check) else amlt_data_dir

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = image + noise
    return noisy_image
def add_salt_and_pepper_noise(image, noise_ratio=0.02):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0] 
        else:
            noisy_image[row, col] = [255, 255, 255]
    return noisy_image
class DataAugmentation:
    def __init__(
        self,
        prob=0.5,
        rotate=True,
        flip=True,
        pixel_shift=True,
        pixel_shift_ratio=0.1,
        crop=True,
        crop_ratio=0.1,
        gaussian_noise=True,
        salt_and_pepper_noise=True,
    ):
        self.prob = prob
        self.rotate = rotate
        self.flip = flip
        self.pixel_shift = pixel_shift
        self.pixel_shift_ratio = pixel_shift_ratio
        self.crop = crop
        self.crop_ratio = crop_ratio
        self.gaussian_noise = gaussian_noise
        self.salt_and_pepper_noise = salt_and_pepper_noise

    def __call__(self, img, mask):
        if random.random() < self.prob:
            # Rotate the image
            if self.rotate:
                rotate_times = random.randint(0, 3)
                img = np.rot90(img, rotate_times, (0, 1))
                mask = np.rot90(mask, rotate_times)
            # Flip the image
            if self.flip:
                flip = random.choice([0, 1, -1])
                img = cv2.flip(img, flip)
                mask = cv2.flip(mask, flip)

            # Crop the image
            if self.crop and random.random() < self.prob:
                h, w = img.shape[:2]
                c = np.array([h, w]) / 2
                # pad the image with zeros
                pad = int(2 * h * self.crop_ratio + 1)
                img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
                mask = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant")

                # crop the image
                scale = 1 + np.random.uniform(-self.crop_ratio, self.crop_ratio)
                new_h, new_w = np.array([h, w]) * scale
                new_c = c + np.random.uniform(
                    -self.crop_ratio, self.crop_ratio, size=2
                ) * np.array([h, w])
                x1, x2 = new_c[0] - new_h / 2, new_c[0] + new_h / 2
                y1, y2 = new_c[1] - new_w / 2, new_c[1] + new_w / 2
                x1 = pad + int(x1)
                x2 = pad + int(x2)
                y1 = pad + int(y1)
                y2 = pad + int(y2)
                img = img[x1:x2, y1:y2, :]
                mask = mask[x1:x2, y1:y2]

                # resize the image
                img = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST)

            # Shift pixel values
            if self.pixel_shift:
                # pixel_max = np.max(img)
                # pixel_min = np.min(img)
                scale = 255.0 #pixel_max - pixel_min
                shift = np.random.uniform(-scale, scale) * self.pixel_shift_ratio
                img = img + shift
                
            # Add Gaussian noise
            if self.gaussian_noise and random.random() < self.prob:
                std = np.random.uniform(0, 25)
                img = add_gaussian_noise(img, mean=0, std=std)
                
            # Add salt and pepper noise
            if self.salt_and_pepper_noise and random.random() < self.prob:
                noise_ratio = np.random.uniform(0.02, 0.1)
                img = add_salt_and_pepper_noise(img, noise_ratio=noise_ratio)

        return img, mask

def choose_prompt(prompts):
    if isinstance(prompts, str):
        return prompts
    elif isinstance(prompts, list):
        return random.choice(prompts)
    else:
        raise ValueError("Invalid prompt type. Must be str or list.")

class BiomedParseDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_prompts=4,
        all_class_masks=False,
        transforms=DataAugmentation(),
        img_size=(512, 512),
        interpolate_mask_size=(512, 512),
        name=None,
        negative=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_prompts = num_prompts
        self.all_class_masks = all_class_masks    # output multiclass mask with all prompts
        self.transforms = transforms
        self.img_size = img_size
        self.interpolate_mask_size = interpolate_mask_size
        self.json_file = os.path.join(root_dir, f"{split}.json")
        self.name = name if name else os.path.basename(os.path.normpath(root_dir))

        with open(self.json_file, "r") as file:
            data = json.load(file)
            self.data_info = data.get("annotations", [])
            self.class_prompts = data.get("class_prompts", None)

        self.images_dir = os.path.join(root_dir, split)
        self.masks_dir = os.path.join(root_dir, f"{split}_mask")
        self.negative = negative

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        ann_info = self.data_info[idx]

        try:
            img_path = os.path.join(self.images_dir, ann_info["file_name"])
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        except (IOError, FileNotFoundError):
            print(f"Image not found: {img_path}")
            image = np.zeros((*self.img_size, 3))
            mask_orig = np.zeros((*self.img_size, 1))

        mask_path = os.path.join(self.masks_dir, ann_info.get("mask_file", ""))
        try:
            mask_orig = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        except (IOError, FileNotFoundError):
            print(f"Mask not found: {mask_path}")
            mask_orig = np.zeros((*self.img_size, 1))
            image = np.zeros((*self.img_size, 3))
        if mask_orig is None:
            mask_orig = np.zeros((*self.img_size, 1))
            image = np.zeros((*self.img_size, 3))
            print(f"Mask is None: {mask_path}")
        # check if mask is binary format with only 0 and 255
        if mask_orig.max() == 255 and len(np.unique(mask_orig)) == 2:
            mask_orig = 0.0 * mask_orig
            image = 0.0 * image
            print(f"Mask is binary format: {mask_path}")
        if len(mask_orig.shape) == 3:
            mask_orig = mask_orig[:, :, 0]

        if self.transforms:
            image, mask_orig = self.transforms(image, mask_orig)
            
        # randomly swap channels 1 and 2
        if random.random() < 0.5:
            image = image[:, :, [0, 2, 1]]
            
        
        # # resize image to 1024x1024
        # image = cv2.resize(
        #     image,
        #     (1024, 1024),
        #     interpolation=cv2.INTER_LINEAR,
        # ).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        
        # prepare prompts and corresponding masks
        if self.class_prompts:
            class_prompts = self.class_prompts
        else:
            class_prompts = ann_info["class_prompts"]
            
        is_instance = ann_info["instance_label"]
            
        if not self.all_class_masks:
            # sample class id
            if is_instance:
                all_classes = [int(_) for _ in class_prompts if _ != 'instance_label']
                class_ids = [all_classes[0] for _ in range(self.num_prompts)]
                mask = mask_orig.astype(np.uint8)
                mask = np.repeat(mask[None, :, :], self.num_prompts, axis=0)
            else:
                all_classes = [int(_) for _ in class_prompts if _ != 'instance_label']
                replace = False if len(all_classes) >= self.num_prompts else True
                class_ids = np.random.choice(all_classes, size=self.num_prompts, replace=replace)
                mask = np.stack(
                    [(1 * (mask_orig == class_id)).astype(np.uint8) for class_id in class_ids]
                )
        else:
            # get one multi class mask with all class ids
            class_ids = [int(_) for _ in class_prompts if _ != 'instance_label']
            class_ids.sort()    # sort class ids
            if is_instance:
                # binary mask for only one class
                mask = (class_ids[0] * (mask_orig > 0)).astype(np.uint8)
            else:
                # multi class mask
                mask = mask_orig.astype(np.uint8)
                               
        # sample prompt
        selected_sentence = [
            choose_prompt(class_prompts[str(class_id)]) for class_id in class_ids
        ]
        selected_sentence = "[SEP]".join(selected_sentence)
        
        class_ids_str = "&".join([str(class_id) for class_id in class_ids])

        return {
            "image": torch.tensor(image.copy(), dtype=torch.float32),
            "labels": torch.tensor(mask.copy(), dtype=torch.long),
            "text": selected_sentence,
            "class_ids": class_ids_str,
            "mask_file": mask_path,
            "instance_label": is_instance,
            "multiclass_label": self.all_class_masks,
        }


