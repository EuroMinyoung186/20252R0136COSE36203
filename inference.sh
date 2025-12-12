#!/bin/bash

export API_KEY="YOUR_KEY_HERE"
export situations="I'm meeting my friends for a casual lunch in the park."

python fashion_recommendation/save.py \
    --img_base_path data/cloth/img_square \
    --json_file data/cloth/test.json \
    --ckpt_path fashion_recommendation/ckpt/fashionrec_step3608.pt

recommendation=$(python llm_generation/inference_rec.py \
            --situation $situations
            --api_key $API_KEY
            )
