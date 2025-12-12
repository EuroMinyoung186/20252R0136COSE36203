# module/FashionRec.py

import torch
import torch.nn as nn

from module.MappingLayer import TransformerMappingNetwork
from module.DescriptionModel import QueryTextEncoder
from module.Image import QueryImageEncoder


class Extraction(nn.Module):
    """
    Query = rec(text), Document = desc(text) + images + mapping(image|desc)
    Forward에서는 바로 in-batch pairwise ColBERT-style score matrix [B, B]를 반환한다.
    """

    def __init__(self):
        super().__init__()
        self.mappingModule = TransformerMappingNetwork()
        self.querytextEncoder = QueryTextEncoder()
        self.queryimageEncoder = QueryImageEncoder()

    def compute_doc_tokens(self, foot, top, pants, desc_batch):
        """
        foot, top, pants: image pixel tensors [B, 3, H, W]
        desc_batch: {"input_ids": ..., "attention_mask": ...}

        Returns:
            doc_tokens: [B, T_doc, D]
        """
        # ---- 1. Extract description features ----
        # desc_feat: [B, T_desc, D]
        desc_feat = self.querytextEncoder(desc_batch)

        # ---- 2. Extract image features ----
        # foot_feat, top_feat, pants_feat: [B, D]
        foot_feat, top_feat, pants_feat = self.queryimageEncoder(foot, top, pants)

        # image tokens: [B, 3, D]
        foot_tok = foot_feat.unsqueeze(1)   # [B, 1, D]
        top_tok = top_feat.unsqueeze(1)     # [B, 1, D]
        pants_tok = pants_feat.unsqueeze(1) # [B, 1, D]
        image_tokens = torch.cat([foot_tok, top_tok, pants_tok], dim=1)  # [B, 3, D]

        # ---- 3. Mapping transformer (image_tokens → conditioned on desc_feat) ----
        # mapping_feat: [B, 3, D]
        mapping_feat = self.mappingModule(image_tokens, desc_feat)

        # ---- 4. Build document tokens ----
        # doc_tokens: [B, 3 + T_desc + 3, D]
        doc_tokens = torch.cat(
            [mapping_feat, desc_feat, image_tokens],
            dim=1
        )
        return doc_tokens


    def forward(self, foot, top, pants, rec, desc):
        doc_tokens = self.compute_doc_tokens(foot, top, pants, desc)
        _, T, _ = doc_tokens.shape

        return doc.tokens
