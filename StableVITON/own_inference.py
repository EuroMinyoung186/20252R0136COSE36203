import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img
from own_data import get_data

# def get_data(human_root_dir, cloth_root_dir, img_H, img_W, cloth_id, cloth_path, img_path,  data_type='test'):

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--human_root_dir", type=str)
    parser.add_argument("--cloth_root_dir", type=str)
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./samples")
    parser.add_argument("--cloth_id", type=str)
    parser.add_argument("--cloth_path", type=str)
    parser.add_argument("--img_path", type=str)

    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    img_H = args.img_H
    img_W = args.img_W

    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params

    model = create_model(config_path=None, config=config)
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    model.load_state_dict(load_cp)
    model = model.cuda()
    model.eval()

    sampler = PLMSSampler(model)

    # def get_data(human_root_dir, cloth_root_dir, img_H, img_W, cloth_id, cloth_path, img_path,  data_type='test'):

    dataset = get_data(
        human_root_dir = args.human_root_dir,
        cloth_root_dir = args.cloth_root_dir,
        img_H = img_H,
        img_W = img_W,
        cloth_id = args.cloth_id,
        cloth_path = args.cloth_path,
        img_path = args.img_path
    )

    shape = (4, img_H//8, img_W//8) 
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    z, c = model.get_input(dataset, params.first_stage_key)
    bs = z.shape[0]
    c_crossattn = c["c_crossattn"][0][:bs]
    if c_crossattn.ndim == 4:
        c_crossattn = model.get_learned_conditioning(c_crossattn)
        c["c_crossattn"] = [c_crossattn]
    uc_cross = model.get_unconditional_conditioning(bs)
    uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
    uc_full["first_stage_cond"] = c["first_stage_cond"]
    for k, v in dataset.items():
        if isinstance(v, torch.Tensor):
            dataset[k] = v.cuda()
    sampler.model.batch = dataset

    ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
    start_code = model.q_sample(z, ts)     

    samples, _, _ = sampler.sample(
        args.denoise_steps,
        bs,
        shape, 
        c,
        x_T=start_code,
        verbose=False,
        eta=args.eta,
        unconditional_conditioning=uc_full,
    )

    x_samples = model.decode_first_stage(samples)
    for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, dataset['img_fn'], dataset["cloth_fn"])):
        x_sample_img = tensor2img(x_sample)  # [0, 255]
        if args.repaint:
            repaint_agn_img = np.uint8((dataset["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
            repaint_agn_mask_img = dataset["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
            x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
            x_sample_img = np.uint8(x_sample_img)

        to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
        cv2.imwrite(to_path, x_sample_img[:,:,::-1])

if __name__ == "__main__":
    args = build_args()
    main(args)
