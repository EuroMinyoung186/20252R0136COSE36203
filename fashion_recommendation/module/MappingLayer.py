import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, q, kv):
        # q: [B, Nq, D]
        # kv: [B, T, D]
        q_norm = self.norm1_q(q)
        kv_norm = self.norm1_kv(kv)

        # Cross Attention: Q from image, K/V from text
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)

        # Residual 1
        x = q + attn_out

        # FFN
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x

class TransformerMappingNetwork(nn.Module):
    def __init__(self, dim=512, num_heads=8, depth=3, mlp_ratio=4.0):
        super().__init__()

        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, img_feat, txt_feat):
        # img_feat: [B, Nq, D]
        # txt_feat: [B, T, D]

        x = img_feat
        for layer in self.layers:
            x = layer(x, txt_feat)

        return x