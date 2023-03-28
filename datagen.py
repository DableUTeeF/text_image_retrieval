from torch.utils.data import Dataset
import torchvision.transforms as transforms
import functools
import os
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import random
import torch
import pandas as pd
import tensorflow as tf


def split(args):
    with open(args.json_path, "rb") as f:
        caption_all = json.load(f)

    group_by_id = dict()
    for record in caption_all:
        if record["id"] not in group_by_id.keys():
            group_by_id[record["id"]] = []
        for caption in record["captions"]:
            group_by_id[record["id"]].append({
                "id": record["id"],
                "file_path": record["file_path"],
                "caption": caption
            })

    train, val = train_test_split([group_by_id[key] for key in group_by_id.keys()], test_size=0.076, random_state=42, shuffle=False)

    return train, val


class JSONDataset(Dataset):
    def __init__(self, args, data, text_model, regen, train):
        self.train = train
        self.args = args
        self.folder = os.path.join(args.text_vector_path, args.text_model)
        self.stage = 'train' if train else 'val'
        self.text_model = text_model
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((args.height, args.width), interpolation=3),
                transforms.Pad(10),
                transforms.RandomCrop((args.height, args.width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((args.height, args.width), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.features = {}
        if regen:
            self.data = functools.reduce(lambda a, b: a + b, data)
            self.data = self.generate()
        else:
            self.data = pd.read_csv(os.path.join(self.folder, f'{self.stage}_indice.csv'))

    @torch.no_grad()
    def generate(self):
        # tokenizer parallelism madness
        if not os.path.isdir(os.path.join(self.folder,  self.stage)):
            os.makedirs(os.path.join(self.folder, self.stage))
        output = {'index': [], 'text': [], 'file': [], 'img': [], 'label': []}
        progbar = tf.keras.utils.Progbar(len(self.data))
        for idx, item in enumerate(self.data):
            txt = self.text_model.encode(item["caption"], convert_to_tensor=True)
            torch.save(txt, os.path.join(self.folder, self.stage, f'{idx:07d}.pth'))
            output['index'].append(idx)
            output['text'].append(item["caption"])
            output['file'].append(f'{idx:07d}.pth')
            output['img'].append(item["file_path"])
            output['label'].append(item["id"])
            progbar.update(idx + 1)
        df = pd.DataFrame(output)
        df.to_csv(os.path.join(self.folder, f'{self.stage}_indice.csv'), index=False)
        return df

    def __getitem__(self, index):
        row = self.data.iloc[index]
        if row['text'] not in self.features:
            txt_vector = torch.load(os.path.join(self.folder, self.stage, row['file']), map_location='cpu')
            self.features[row['text']] = txt_vector
        label = row['label']
        img = Image.open(os.path.join(self.args.image_root_path, row["img"]))
        img = self.transform(img)
        return img, self.features[row['text']], label

    def __len__(self):
        return len(self.data)


class SiameseDataset(Dataset):  # todo: refactory nightmare
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, text, label_1 = self.data[index]
        label_2 = label_1
        if random.random() > 0.5:  # todo: try triplet output with self.data.data[self.data.data['label']==label_1]
            _, text, label_2 = self.data[random.randint(0, len(self.data)-1)]
        return img, text, (label_1, label_2)

