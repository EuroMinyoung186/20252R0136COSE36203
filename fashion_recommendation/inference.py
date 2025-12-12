# train_ddp.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import argparse
from module.FashionRec import FashionRec
from transformers import BertTokenizer, BertModel



def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendation", type=str, required=True)
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()

def inference(args):
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FashionRec().to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    db = torch.load(args.database_path, map_location=device)

    all_ids = db["ids"]
    all_doc_embs = db["doc_embs"]

    tokenized_query = tokenizer(
        args.recommendation,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    tokenized_query = {
        k: v.to(device) for k, v in tokenized_query.items()
    }

    with torch.no_grad():
        query_emb = model.encode_query(tokenized_query)  # [1, D]
        query_emb = F.normalize(query_emb, dim=1)

    sims = query_emb @ all_doc_embs.T
    topk = torch.topk(sims, k=1, dim=1)

    top_indices = topk.indices[0].tolist()
    top_scores = topk.values[0].tolist()

    top_ids = [all_ids[i] for i in top_indices]

    with open(args.save_path, "w", encoding="utf-8") as f:
        for rank, (img_id, score) in enumerate(zip(top_ids, top_scores), start=1):
            f.write(f"{rank}\t{img_id}\t{score:.4f}\n")

    print(f"✅ Saved Top-5 recommendations to {args.save_path}")
    


def main():
    args = build_args()
    inference(args)


if __name__ == "__main__":
    main()
