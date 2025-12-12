import os
import torch
import random
from os.path import join as opj

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None
):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = cv2.imread(p)
    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img
def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img

def get_data(human_root_dir, cloth_root_dir, img_H, img_W, cloth_id, cloth_path, img_path,  data_type='test'):

    with open(cloth_id, 'r') as f:
        lines = f.readlines()
    length = len(lines)
    choice = random.randint(0, length - 1)
    cloth_id = lines[choice].split('\t')[1]


    resize_ratio_H = 1.0
    resize_ratio_W = 1.0
    resize_transform = A.Resize(img_H, img_W)
    transform_size_lst = [A.Resize(int(img_H*resize_ratio_H), int(img_W*resize_ratio_W))]

    transform_size = A.Compose(
        transform_size_lst,
        additional_targets={"agn":"image", 
                            "agn_mask":"image", 
                            "cloth":"image", 
                            "cloth_mask":"image", 
                            "cloth_mask_warped":"image", 
                            "cloth_warped":"image", 
                            "image_densepose":"image", 
                            "image_parse":"image", 
                            "gt_cloth_warped_mask":"image",
                            }
    )
    image = imread_for_albu(opj(human_root_dir, data_type, "image", img_path))
    H, W, _ = image.shape
    agn = imread_for_albu(opj(human_root_dir, data_type, "agnostic-v3.2", img_path))
    agn_mask = imread_for_albu(opj(human_root_dir, data_type, "agnostic-mask", img_path.replace(".jpg", "_mask.png")), is_mask=True)
    cloth = imread_for_albu(
        opj(cloth_root_dir, "img_square", cloth_id, cloth_path),
        use_resize=True,
        height=H,
        width=W,
    )

    cloth_mask = imread_for_albu(
        opj(cloth_root_dir, "mask_square", cloth_id, cloth_path.replace('.jpg', '.png')),
        is_mask=True,
        cloth_mask_check=True,
        use_resize=True,
        height=H,
        width=W,
    )
        
    gt_cloth_warped_mask = np.zeros_like(agn_mask)
        
    
    image_densepose = imread_for_albu(opj(human_root_dir, data_type, "image-densepose", img_path))

    

    if transform_size is not None:
        transformed = transform_size(
            image=image, 
            agn=agn, 
            agn_mask=agn_mask, 
            cloth=cloth, 
            cloth_mask=cloth_mask, 
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
        )
        image=transformed["image"]
        agn=transformed["agn"]
        agn_mask=transformed["agn_mask"]
        image_densepose=transformed["image_densepose"]
        gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]

        cloth=transformed["cloth"]
        cloth_mask=transformed["cloth_mask"]
        
    

    agn_mask = 255 - agn_mask
    agn = norm_for_albu(agn)
    agn_mask = norm_for_albu(agn_mask, is_mask=True)
    cloth = norm_for_albu(cloth)
    cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
    image = norm_for_albu(image)
    image_densepose = norm_for_albu(image_densepose)
    gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)

    agn = torch.from_numpy(agn).unsqueeze(0)
    agn_mask = torch.from_numpy(agn_mask).unsqueeze(0)

    cloth = torch.from_numpy(cloth).unsqueeze(0)
    cloth_mask = torch.from_numpy(cloth_mask).unsqueeze(0)

    image = torch.from_numpy(image).unsqueeze(0)
    image_densepose = torch.from_numpy(image_densepose).unsqueeze(0)

    gt_cloth_warped_mask = torch.from_numpy(gt_cloth_warped_mask).unsqueeze(0)
        
    return dict(
        agn=agn,
        agn_mask=agn_mask,
        cloth=cloth,
        cloth_mask=cloth_mask,
        image=image,
        image_densepose=image_densepose,
        gt_cloth_warped_mask=gt_cloth_warped_mask,
        txt="",
        img_fn=img_path,
        cloth_fn=cloth_path
    )