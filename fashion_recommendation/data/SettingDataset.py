import os
import json
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer

class SettingDataset(Dataset):
    def __init__(
        self, 
        img_base_path="/home/.../mask_square",
        json_file="/home/.../temp_sit2rec_output.json",
    ):
        with open(json_file, 'r') as f:
            dataframe = json.load(f)

        self.img_files = []
        self.keys = []
        self.description = []

        valid_items = ['footwear.jpg', 'pants.jpg', 'top.jpg']

        for img_file, value in dataframe.items():
            img_dir_name = img_file.split('.')[0]
            img_dir = os.path.join(img_base_path, img_dir_name)

            if not os.path.exists(img_dir):
                continue

            clothes = os.listdir(img_dir)
            if len(clothes) < 3:
                continue

            tmp_paths = {}

            for cloth in clothes:
                if cloth in valid_items:
                    tmp_paths[cloth.replace(".jpg","")] = os.path.join(img_dir, cloth)

            if len(tmp_paths) != 3:
                continue

            self.img_files.append(tmp_paths)
            self.description.append(value["caption"])
            self.keys.append(img_dir_name)

        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 텍스트
        desc = self.description[idx]
        cid = self.keys[idx]

        # raw text만 반환 (tokenize는 collate_fn 또는 trainer에서)
        # 이미지도 PIL Image 그대로 반환
        imgs = {}
        paths = self.img_files[idx]
        for key, path in paths.items():
            try:
                imgs[key] = Image.open(path).convert("RGB")
            except:
                imgs[key] = Image.new("RGB", (256,256))
        
        return {
            "description": desc,
            "footwear": imgs["footwear"],
            "pants": imgs["pants"],
            "top": imgs["top"],
            "id": cid
        }
