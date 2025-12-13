import os
import json
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer

class TestDataset(Dataset):
    def __init__(
        self, 
        json_file="/home/.../temp_sit2rec_output.json",
    ):
        with open(json_file, 'r') as f:
            dataframe = json.load(f)

        self.img_id = []
        self.recommendations = []

        for img_file, value in dataframe.items():
            self.img_id.append(img_file.split('.')[0])
            self.recommendations.append(value["recommendation"])

    def __len__(self):
        return len(self.recommendations)

    def __getitem__(self, idx):
        rec = self.recommendations[idx]
        img_id = self.img_id[idx]

        return {
            "id" : img_id,
            "recommendation": rec
        }
