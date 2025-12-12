import torch
import torch.nn as nn 
from transformers import BertModel

class QueryTextEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.proj = nn.Linear(self.model.config.hidden_size, out_dim)  # 768 -> 512
        # 선택: 안정성 위해 LayerNorm 하나 더 둘 수도 있음
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        outputs = self.model(**x)                          # last_hidden_state: (B, T, 768)
        token_embeddings = outputs.last_hidden_state       # (B, T, 768)

        # 768 -> 512 projection
        token_embeddings = self.proj(token_embeddings)     # (B, T, 512)
        token_embeddings = self.norm(token_embeddings)     # (B, T, 512)

        return token_embeddings