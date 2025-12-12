#bin/bash

python fashion_recommendation/save.py \
    --img_base_path data/cloth/img_square \
    --json_file data/cloth/test.json \
    --ckpt_path fashion_recommendation/ckpt/fashionrec_step3608.pt