export db_path="data/cloth/doc_toens_db.pt"
export ckpt_path="fashion_recommendation/ckpt/ckpt.pt"
export save_path="data/tmp.txt"

python fashion_recommendation/test.py \
    --json_file "data/cloth/test.json" \
    --database_path $db_path \
    --ckpt_path $ckpt_path \
    --save_path $save_path