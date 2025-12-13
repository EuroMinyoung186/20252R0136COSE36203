#!/bin/bash

export situations="I'm heading to a casual weekend brunch with friends."
export recommendation="Pair your chiffon long-sleeve shirt with a lapel neckline with medium-length chiffon pants in a solid color for a chic and comfortable outfit perfect for a casual brunch with friends on a sunny afternoon. Opt for cute sandals or flats to complete the look."
export db_path="data/cloth/doc_toens_db.pt"
export ckpt_path="fashion_recommendation/ckpt/ckpt.pt"
export save_path="data/tmp.txt"

python fashion_recommendation/save.py \
    --img_base_path data/cloth/img_square \
    --json_file data/cloth/test.json \
    --ckpt_path $ckpt_path \
    --save_path $db_path

python fashion_recommendation/inference.py \
    --recommendation "$recommendation" \
    --database_path $db_path \
    --ckpt_path $ckpt_path \
    --save_path $save_path

python StableVITON/own_inference.py \
    --config_path "StableVITON/configs/VITONHD.yaml" \
    --model_load_path "StableVITON/ckpt/VITONHD.ckpt" \
    --human_root_dir "data/human" \
    --cloth_root_dir "data/cloth" \
    --cloth_id $save_path \
    --cloth_path "top.jpg" \
    --img_path "11001_00.jpg"

