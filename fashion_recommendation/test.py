import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from module.FashionRec import FashionRec


# ======================
# Dataset
# ======================

class FigureDataset(Dataset):
    def __init__(self, path, limit=1000):
        with open(path, "r") as f:
            datas = json.load(f)

        keys = list(datas.keys())[:limit]

        self.captions = []
        self.recs = []

        for k in keys:
            self.captions.append(datas[k]['caption'])
            self.recs.append(datas[k]['recommendation'])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx], self.recs[idx]


def collate_fn(batch):
    captions, recs = zip(*batch)
    captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    recs = tokenizer(recs, padding=True, truncation=True, return_tensors="pt")
    return captions, recs


# ======================
# Load trained model
# ======================

ckpt_path = "/home/aikusrv02/autonomous_driving_ai_challenge/machine_learning/FashionRecommendation/src/fashionrec_step3608.pt"
device = "cuda"

model = FashionRec()
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()

# Extract encoders
cap_encoder = model.querytextEncoder      # caption encoder
rec_encoder = model.documentEncoder       # recommendation encoder

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

path = "/home/aikusrv02/autonomous_driving_ai_challenge/machine_learning/FashionRecommendation/data/trainset.json"

dataset = FigureDataset(path, limit=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# ======================
# Extract embeddings
# ======================

caption_embeds = []
rec_embeds = []

with torch.no_grad():
    for captions, recs in dataloader:

        captions = {k: v.to(device) for k, v in captions.items()}
        recs = {k: v.to(device) for k, v in recs.items()}

        # caption: [B, T, D]
        cap_feat = cap_encoder(captions)     # (B, T, D)
        cap_feat = cap_feat.mean(dim=1)      # mean pooling → (B, D)

        # recommendation: [B, T, D]
        rec_feat = rec_encoder(recs)         # (B, T, D)
        rec_feat = rec_feat.mean(dim=1)      # mean pooling → (B, D)

        caption_embeds.append(cap_feat.cpu())
        rec_embeds.append(rec_feat.cpu())

caption_embeds = torch.cat(caption_embeds, dim=0)
rec_embeds = torch.cat(rec_embeds, dim=0)

print("caption_embeds:", caption_embeds.shape)
print("rec_embeds:", rec_embeds.shape)


# ======================
# t-SNE Visualization (Save Image)
# ======================

all_embeds = torch.cat([caption_embeds, rec_embeds], dim=0).numpy()
labels = np.array([0] * len(caption_embeds) + [1] * len(rec_embeds))

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
proj = tsne.fit_transform(all_embeds)

plt.figure(figsize=(10, 10))
plt.scatter(proj[labels==0, 0], proj[labels==0, 1], alpha=0.6, s=10, label="Captions")
plt.scatter(proj[labels==1, 0], proj[labels==1, 1], alpha=0.6, s=10, label="Recommendations")

plt.legend()
plt.title("t-SNE of Caption vs Recommendation Embeddings (from Trained Model)")

out_path = "tsne_trained_model.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved scatter plot to: {out_path}")
