import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from ..utils import process_input
# Determine the data directory dynamically
# amlt_data_dir = os.getenv("AMLT_DATA_DIR", "/mnt/default/data")
# path_to_check = "/mnt/external/data"
# DATA_DIR = path_to_check if os.path.exists(path_to_check) else amlt_data_dir

def get_axis(img):
    # get the axis to slice the 3D volume
    shape = img.shape
    # get shape difference between the axes
    diff_ratio = [2*abs(shape[1]-shape[2])/(shape[1]+shape[2]),
            2*abs(shape[0]-shape[2])/(shape[0]+shape[2]),
            2*abs(shape[0]-shape[1])/(shape[0]+shape[1])]
    
    if diff_ratio[0] < 0.5:
        valid_axis = 0
    else:
        min_axis = np.argmin(shape)
        valid_axis = min_axis
        
    return valid_axis

def choose_prompt(prompts):
    if isinstance(prompts, str):
        return prompts
    elif isinstance(prompts, list):
        return random.choice(prompts)
    else:
        raise ValueError("Invalid prompt type. Must be str or list.")
    


class BiomedParseDataset3D(Dataset):
    def __init__(
        self,
        root_dir,
        img_size=(512, 512),
        interpolate_mask_size=(512, 512),
        name=None,
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.interpolate_mask_size = interpolate_mask_size
        self.name = name if name else os.path.basename(os.path.normpath(root_dir))
        self.images_dir = root_dir
            
        self.file_list = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        
        try:
            # load the 3D volume
            img_path = os.path.join(self.images_dir, file)
            data = np.load(img_path, allow_pickle=True)
            image = data['imgs'].astype(np.uint8)
            mask_path = img_path.replace("/test/", "/test_gt/")
            data_gt = np.load(mask_path)
            mask = data_gt['gts'].astype(np.uint8)
            class_prompts = data['text_prompts'].item()
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
            return []
        
        image, pad_width, padded_size, valid_axis = process_input(image, self.img_size[0])
        
        # prepare prompts and corresponding masks
        is_instance = class_prompts["instance_label"]
            
        # get one multi class mask with all class ids
        class_ids = [int(_) for _ in class_prompts if _ != 'instance_label']
        class_ids.sort()    # sort class ids
        
        if is_instance:
            # binary mask for only one class
            mask = (class_ids[0] * (mask > 0)).astype(np.uint8)
        else:
            # multi class mask
            mask = mask.astype(np.uint8)
                               
        # sample prompt
        selected_sentence = [
            choose_prompt(class_prompts[str(class_id)]) for class_id in class_ids
        ]
        selected_sentence = "[SEP]".join(selected_sentence)
        
        class_ids_str = "&".join([str(class_id) for class_id in class_ids])

        return {
            "image": image.to(dtype=torch.uint8),
            "labels": torch.tensor(mask, dtype=torch.uint8),
            "text": selected_sentence,
            "class_ids": class_ids_str,
            "mask_file": mask_path,
            "instance_label": is_instance,
            "axis": valid_axis,
            "pad_width": torch.tensor(pad_width, dtype=torch.int64),
            "padded_size": padded_size,
        }


