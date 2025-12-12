import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.utils import collate_fn_sample
from module.FashionRec import FashionRec
from data.SettingDataset import SettingDataset


def build_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_base_path", type=str, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="doc_emb_db.pt")
    return parser.parse_args()


def main():
    args = build_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------
    # 1) Load Model
    # ------------------------------------
    model = FashionRec().to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ------------------------------------
    # 2) Dataset & Dataloader
    # ------------------------------------
    dataset = SettingDataset(
        img_base_path=args.img_base_path,
        json_file=args.json_file,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,      # document embedding은 batch=1이 가장 안전
        num_workers=4,
        collate_fn=collate_fn_sample,
        pin_memory=True,
        shuffle=False,
    )

    all_ids = []
    all_embs = []

    # ------------------------------------
    # 3) Extract and Save document embeddings
    # ------------------------------------
    for batch in tqdm(dataloader, desc="Extracting document embeddings"):

        img_id = batch["id"][0]

        desc_batch = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }

        foot = batch["foot_pixel"].to(device)
        pants = batch["pants_pixel"].to(device)
        top = batch["top_pixel"].to(device)

        with torch.no_grad():
            doc_emb = model.encode_document(
                foot=foot,
                top=top,
                pants=pants,
                desc_batch=desc_batch,
            )   # [1, D]

            doc_emb = torch.nn.functional.normalize(doc_emb, dim=1)

        doc_emb = doc_emb.squeeze(0).cpu()   # [D]

        all_ids.append(img_id)
        all_embs.append(doc_emb)

    # ------------------------------------
    # 4) Save
    # ------------------------------------
    save_obj = {
        "ids": all_ids,
        "doc_embs": torch.stack(all_embs, dim=0)  # [N, D]
    }

    torch.save(save_obj, args.save_path)
    print(f"\nSaved {len(all_ids)} document embeddings → {args.save_path}")


if __name__ == "__main__":
    main()
