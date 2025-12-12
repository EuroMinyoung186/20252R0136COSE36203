import torch
import torch.nn as nn
from transformers import CLIPModel

class QueryImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")

    def forward(self, foot_pixel, top_pixel, pants_pixel):
        foot_feat = self.model.get_image_features(pixel_values=foot_pixel)
        top_feat = self.model.get_image_features(pixel_values=top_pixel)
        pants_feat = self.model.get_image_features(pixel_values=pants_pixel)
        return foot_feat, top_feat, pants_feat

