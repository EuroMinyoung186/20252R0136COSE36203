#!/bin/bash

export situations="I'm meeting my friends for a casual lunch in the park."
export recommendation="Wear your graphic-patterned cotton crew neck top with your solid long denim bottoms for a stylish yet casual lunch outfit. Complete the look with your wrist accessory and ring for a touch of personality, and opt for comfortable flats or sneakers to enjoy your time in the park with friends."
export db_path="data/cloth/doc_toens_db.pt"
export ckpt_path="fashion_recommendation/ckpt/ckpt.pt"
export save_path="data/tmp.txt"

#python fashion_recommendation/save.py \
#    --img_base_path data/cloth/img_square \
#    --json_file data/cloth/test.json \
#    --ckpt_path $ckpt_path \
#    --save_path $db_path

#python fashion_recommendation/inference.py \
#    --recommendation "$recommendation" \
#    --database_path $db_path \
#    --ckpt_path $ckpt_path \
#    --save_path $save_path

python StableVITON/own_inference.py \
    --config_path "StableVITON/configs/VITONHD.yaml" \
    --model_load_path "StableVITON/ckpt/VITONHD.ckpt" \
    --human_root_dir "data/human" \
    --cloth_root_dir "data/cloth" \
    --cloth_id $save_path \
    --cloth_path "top.jpg" \
    --img_path "11001_00.jpg"

