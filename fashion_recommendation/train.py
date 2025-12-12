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

from data.TrainDataset import CustomDataset
from utils.utils import collate_fn
from module.FashionRec import FashionRec


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_base_path", type=str, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


def train(
    img_base_path,
    json_file,
    batch_size=8,
    num_epochs=20,
    lr=1e-4,
    weight_decay=1e-2,
    num_workers=4,
    save_path="fashionrec.pt",
    temperature=0.07,
):

    # -------------------------
    # DDP basic info
    # -------------------------
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"[Rank {rank}] world_size={world_size}, device={device}")

    # -------------------------
    # Dataset / Sampler
    # -------------------------
    dataset = CustomDataset(
        img_base_path=img_base_path,
        json_file=json_file,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------
    # Model / Optimizer
    # -------------------------
    model = FashionRec().to(device)

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ==========================
    # Training Loop
    # ==========================
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()

        total_loss = 0.0
        total_batches = 0

        data_iter = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Rank {rank}]",
            disable=(rank != 0),
        )

        for batch in data_iter:

            # -------------------------
            # batch → device
            # -------------------------
            desc_batch = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            }
            rec_batch = {
                "input_ids": batch["rec_ids"].to(device, non_blocking=True),
                "attention_mask": batch["rec_mask"].to(device, non_blocking=True),
            }

            foot = batch["foot_pixel"].to(device, non_blocking=True)
            pants = batch["pants_pixel"].to(device, non_blocking=True)
            top = batch["top_pixel"].to(device, non_blocking=True)

            B_local = foot.size(0)

            # -------------------------
            # Forward (CLS embeddings)
            # -------------------------
            query_emb, doc_emb = model(
                foot=foot,
                top=top,
                pants=pants,
                rec=rec_batch,
                desc=desc_batch,
            )
            # query_emb: [B_local, D]
            # doc_emb:   [B_local, D]

            # L2 normalize (CLIP-style, 강력 추천)
            query_emb = F.normalize(query_emb, dim=1)
            doc_emb = F.normalize(doc_emb, dim=1)

            # -------------------------
            # Global gather (autograd-enabled)
            # -------------------------
            query_all = torch.cat(
                dist_fn.all_gather(query_emb), dim=0
            )  # [B_global, D]
            doc_all = torch.cat(
                dist_fn.all_gather(doc_emb), dim=0
            )  # [B_global, D]

            # -------------------------
            # Similarity logits
            # -------------------------
            logits_q = (query_emb @ doc_all.t()) / temperature
            logits_d = (doc_emb @ query_all.t()) / temperature

            # -------------------------
            # Labels (global index)
            # -------------------------
            labels = torch.arange(B_local, device=device) + rank * B_local

            # -------------------------
            # Symmetric contrastive loss
            # -------------------------
            loss_q = F.cross_entropy(logits_q, labels)
            loss_d = F.cross_entropy(logits_d, labels)
            loss = 0.5 * (loss_q + loss_d)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if rank == 0:
                data_iter.set_postfix(loss=f"{total_loss / total_batches:.4f}")
                print(logits_q[0])

        # -------------------------
        # Epoch end logging / ckpt
        # -------------------------
        if rank == 0:
            avg_loss = total_loss / max(total_batches, 1)
            print(f"[Epoch {epoch+1}/{num_epochs}] mean loss = {avg_loss:.4f}")

            base, ext = os.path.splitext(save_path)
            ckpt_path = f"{base}_epoch{epoch+1}{ext}"
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Rank 0] Saved checkpoint -> {ckpt_path}")

    # -------------------------
    # Final save
    # -------------------------
    if rank == 0:
        torch.save(model.module.state_dict(), save_path)
        print(f"[Rank 0] Saved final model -> {save_path}")


def main():
    dist.init_process_group(backend="nccl")

    args = build_args()

    try:
        train(
            img_base_path=args.img_base_path,
            json_file=args.json_file,
            batch_size=6,
            num_epochs=20,
            lr=1e-4,
            weight_decay=1e-2,
            num_workers=4,
            save_path=args.save_path,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
