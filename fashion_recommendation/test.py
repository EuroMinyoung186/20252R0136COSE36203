# test_retrieval.py

import os
import json
import argparse
from typing import Any, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.TestDataset import TestDataset
from utils.utils import collate_fn_test
from module.FashionRec import FashionRec


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--database_path", type=str, required=True)  # torch.load: {"ids", "doc_embs"}
    p.add_argument("--json_file", type=str, required=True)      # TestDataset input
    p.add_argument("--ckpt_path", type=str, default=None)       # (선택) 모델 weight 로드
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_path", type=str, default=None)       # (선택) 예측 결과 저장(jsonl)
    return p.parse_args()


def _is_tensor_ids(x: Any) -> bool:
    return torch.is_tensor(x)


def _gather_ids(all_ids: Union[List[Any], torch.Tensor], idx: torch.Tensor) -> Union[List[List[Any]], torch.Tensor]:
    """
    idx: [B, K] (cpu or cuda)
    returns:
      - if all_ids is Tensor: Tensor [B, K]
      - if all_ids is list:  python list of lists length B, each length K
    """
    if _is_tensor_ids(all_ids):
        # all_ids: [N]
        all_ids_t = all_ids.to(idx.device)
        return all_ids_t[idx]  # [B, K]
    else:
        # list indexing
        idx_cpu = idx.detach().cpu().tolist()
        return [[all_ids[j] for j in row] for row in idx_cpu]


def _to_cid_list(cid: Any, batch_size: int) -> List[Any]:
    """
    cid could be:
      - list/tuple of ids length B
      - tensor of shape [B]
      - single scalar / string when B==1
    """
    if isinstance(cid, (list, tuple)):
        return list(cid)
    if torch.is_tensor(cid):
        cid = cid.detach().cpu().tolist()
        return cid if isinstance(cid, list) else [cid]
    # scalar / str
    return [cid] if batch_size == 1 else list(cid)


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TestDataset(json_file=args.json_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_test,
    )

    # --- Model ---
    model = FashionRec().to(device).eval()
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # ckpt 포맷이 {"state_dict": ...} 인지 그냥 state_dict 인지에 따라 처리
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)

    # --- DB ---
    db = torch.load(args.database_path, map_location="cpu")
    all_ids = db["ids"]          # list[str] or Tensor[N]
    all_doc_embs = db["doc_embs"]  # Tensor[N, D]

    # normalize + move to device
    all_doc_embs = all_doc_embs.to(device, non_blocking=True)
    all_doc_embs = F.normalize(all_doc_embs, dim=1)

    K = args.topk
    total = 0
    hit1 = 0
    hitk = 0

    out_f = open(args.save_path, "w", encoding="utf-8") if args.save_path else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testing (topk={K})"):
            cid = batch["id"]  # GT id (B,)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            query_emb = model.encode_query(input_ids=input_ids, attention_mask=attention_mask)  # [B, D]
            query_emb = F.normalize(query_emb, dim=1)

            sims = query_emb @ all_doc_embs.t()  # [B, N]

            topk_vals, topk_idx = sims.topk(K, dim=1, largest=True, sorted=True)  # [B,K]
            topk_ids = _gather_ids(all_ids, topk_idx)

            B = sims.size(0)
            gt_list = _to_cid_list(cid, B)

            # hits + top1
            if torch.is_tensor(topk_ids):
                # topk_ids: Tensor[B,K]
                gt_t = torch.tensor(gt_list, device=topk_ids.device) if not torch.is_tensor(cid) else cid.to(topk_ids.device)
                gt_t = gt_t.view(-1, 1)
                hits = (topk_ids == gt_t).any(dim=1)  # [B]
                pred1 = topk_ids[:, 0]                # [B]
                hit1_batch = (pred1 == gt_t.squeeze(1)).sum().item()
                hitk_batch = hits.sum().item()
                pred1_list = pred1.detach().cpu().tolist()
                topk_list = topk_ids.detach().cpu().tolist()
            else:
                # topk_ids: List[List[id]]
                hits_bool = [gt in cand for gt, cand in zip(gt_list, topk_ids)]
                hitk_batch = sum(hits_bool)
                hit1_batch = sum(1 for gt, cand in zip(gt_list, topk_ids) if cand[0] == gt)
                pred1_list = [cand[0] for cand in topk_ids]
                topk_list = topk_ids

            total += B
            hit1 += hit1_batch
            hitk += hitk_batch

            # (선택) 저장
            if out_f is not None:
                # 각 샘플별로 top1/topk 저장
                for i in range(B):
                    rec = {
                        "gt": gt_list[i],
                        "pred_top1": pred1_list[i],
                        "pred_topk": topk_list[i],
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if out_f is not None:
        out_f.close()

    r1 = hit1 / max(total, 1)
    rk = hitk / max(total, 1)
    print(f"Total={total} | Recall@1={r1:.4f} | Recall@{K}={rk:.4f}")
    if args.save_path:
        print(f"Saved predictions to: {args.save_path}")


def main():
    args = build_args()
    test(args)


if __name__ == "__main__":
    main()
