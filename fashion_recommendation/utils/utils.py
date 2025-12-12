import torch
from transformers import BertTokenizer
import torchvision.transforms as T

# BERT tokenizer (그대로 사용)
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# CLIP/Fashion-CLIP에서 쓰는 mean/std
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# 256x256 + CLIP 정규화
clip_transform_256 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),                      # -> (3, H, W), [0,1]
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def collate_fn(batch):
    # ------------ TEXT ------------
    descriptions = [item["description"] for item in batch]
    recommendations = [item["recommendation"] for item in batch]

    encoded_desc = tokenizer(
        descriptions,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded_rec = tokenizer(
        recommendations,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    # ------------ IMAGES ------------
    foot_imgs = [item["footwear"] for item in batch]  # PIL.Image (RGB)
    pants_imgs = [item["pants"] for item in batch]
    top_imgs = [item["top"] for item in batch]

    # 직접 256x256 + normalize 적용
    foot_pixel = torch.stack([clip_transform_256(img) for img in foot_imgs], dim=0)   # (B, 3, 256, 256)
    pants_pixel = torch.stack([clip_transform_256(img) for img in pants_imgs], dim=0)
    top_pixel = torch.stack([clip_transform_256(img) for img in top_imgs], dim=0)

    return {
        "input_ids": encoded_desc["input_ids"],
        "attention_mask": encoded_desc["attention_mask"],

        "rec_ids": encoded_rec["input_ids"],
        "rec_mask": encoded_rec["attention_mask"],

        "foot_pixel": foot_pixel,
        "pants_pixel": pants_pixel,
        "top_pixel": top_pixel,
    }


def collate_fn_sample(batch):
    # ------------ TEXT ------------
    descriptions = [item["description"] for item in batch]
    
    encoded_desc = tokenizer(
        descriptions,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )


    # ------------ IMAGES ------------
    cid = [item["id"] for item in batch]
    foot_imgs = [item["footwear"] for item in batch]  # PIL.Image (RGB)
    pants_imgs = [item["pants"] for item in batch]
    top_imgs = [item["top"] for item in batch]

    # 직접 256x256 + normalize 적용
    foot_pixel = torch.stack([clip_transform_256(img) for img in foot_imgs], dim=0)   # (B, 3, 256, 256)
    pants_pixel = torch.stack([clip_transform_256(img) for img in pants_imgs], dim=0)
    top_pixel = torch.stack([clip_transform_256(img) for img in top_imgs], dim=0)

    return {
        "input_ids": encoded_desc["input_ids"],
        "attention_mask": encoded_desc["attention_mask"],

        "foot_pixel": foot_pixel,
        "pants_pixel": pants_pixel,
        "top_pixel": top_pixel,

        "id" : cid
    }