#!/bin/bash

export situations="I'm meeting my friends for a casual lunch in the park."
export recommendation="Wear your graphic-patterned cotton crew neck top with your solid long denim bottoms for a stylish yet casual lunch outfit. Complete the look with your wrist accessory and ring for a touch of personality, and opt for comfortable flats or sneakers to enjoy your time in the park with friends."

python fashion_recommendation/save.py \
    --img_base_path data/cloth/img_square \
    --json_file data/cloth/test.json \
    --ckpt_path fashion_recommendation/ckpt/fashionrec_step3608.pt