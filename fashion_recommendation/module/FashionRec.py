# module/FashionRec.py

import torch
import torch.nn as nn

from module.MappingLayer import TransformerMappingNetwork
from module.DescriptionModel import QueryTextEncoder
from module.RecommendationModule import DocumentEncoder
from module.Image import QueryImageEncoder


class FashionRec(nn.Module):
    """
    Query = rec(text)
    Document = desc(text) + images + mapping(image|desc)
    Document side는 CLS 기반 Transformer encoder로 요약
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # 기존 모듈들
        self.mappingModule = TransformerMappingNetwork()
        self.querytextEncoder = QueryTextEncoder()
        self.queryimageEncoder = QueryImageEncoder()
        self.documentEncoder = DocumentEncoder()   # rec(text) encoder

        # -------------------------
        # Document Transformer (CLS-based)
        # -------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,   # 중요: [B, T, D]
            norm_first=True,
        )
        self.doc_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.doc_norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------
    # Document side
    # ------------------------------------------------
    def compute_doc_tokens(self, foot, top, pants, desc_batch):
        """
        Returns:
            doc_tokens: [B, T_doc, D]
        """
        # 1. description tokens
        desc_feat = self.querytextEncoder(desc_batch)   # [B, T_desc, D]

        # 2. image features
        foot_feat, top_feat, pants_feat = self.queryimageEncoder(foot, top, pants)
        image_tokens = torch.stack(
            [foot_feat, top_feat, pants_feat], dim=1
        )  # [B, 3, D]

        # 3. mapping (image conditioned on desc)
        mapping_feat = self.mappingModule(image_tokens, desc_feat)  # [B, 3, D]

        # 4. concat all tokens
        doc_tokens = torch.cat(
            [mapping_feat, desc_feat, image_tokens],
            dim=1
        )  # [B, T_doc, D]

        return doc_tokens

    def encode_document(self, foot, top, pants, desc_batch):
        """
        Returns:
            doc_cls: [B, D]
        """
        B = foot.size(0)

        doc_tokens = self.compute_doc_tokens(foot, top, pants, desc_batch)
        # doc_tokens: [B, T_doc, D]

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, D]
        tokens = torch.cat([cls, doc_tokens], dim=1)  # [B, 1+T_doc, D]

        # transformer
        out = self.doc_transformer(tokens)  # [B, 1+T_doc, D]

        # CLS pooling
        doc_cls = self.doc_norm(out[:, 0])  # [B, D]

        return doc_cls

    # ------------------------------------------------
    # Query side
    # ------------------------------------------------
    def encode_query(self, rec_batch):
        """
        Returns:
            query_feat: [B, D]
        """
        # 기존 DocumentEncoder가 token을 반환한다면
        # CLS만 쓰는 게 안정적
        rec_tokens = self.documentEncoder(rec_batch)  # [B, T_rec, D]
        query_feat = rec_tokens[:, 0]                 # CLS
        return query_feat

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------
    def forward(self, foot, top, pants, rec, desc):
        """
        Returns:
            query_emb: [B, D]
            doc_emb:   [B, D]
        """
        query_emb = self.encode_query(rec)                       # [B, D]
        doc_emb = self.encode_document(foot, top, pants, desc)  # [B, D]

        return query_emb, doc_emb
