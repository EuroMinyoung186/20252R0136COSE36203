#!/bin/bash

export API_KEY="YOUR_KEY_HERE"

python llm_generation/situation_generate.py \
    --api_key $API_KEY \
    --caption_path data/cloth/captions.json \
    --output_path data/cloth/situations.json

python llm_generation/sit2rec.py \
    --api_key $API_KEY \
    --situation_path data/cloth/situations.json \
    --caption_path data/cloth/captions.json \
    --output_path data/cloth/rec.json