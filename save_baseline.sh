#bin/bash

python fashion_recommendation/save_baseline.py \
    --img_base_path "data/cloth/img_square" \
    --json_file "data/cloth/test.json" \
    --save_path "data/cloth/doc_toens_db_baseline.pt"