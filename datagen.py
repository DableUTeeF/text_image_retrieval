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
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sentence_transformers import InputExample
from timm.data import create_transform

__all__ = ['split', 'JSONDataset', 'SiameseDataset',
           'CMPMDataset', 'TripLetDataset', 'SentenseDataset']

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

def build_transform(is_train, args):
    if is_train:
        transform = create_transform(
            input_size=384,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
        transform.transforms[0] = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=Image.BICUBIC),
            transforms.Pad(10),
            transforms.RandomCrop((args.height, args.width)),
        ])
        return transform

    t = [transforms.Resize((args.height, args.width), interpolation=Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]

    return transforms.Compose(t)


class JSONDataset(Dataset):
    def __init__(self, args, data, text_model, regen, train):
        self.train = train
        self.args = args
        self.folder = os.path.join(args.text_vector_path, args.text_model)
        self.stage = 'train' if train else 'val'
        self.text_model = text_model
        self.transform = build_transform(train, args)
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


class SiameseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, text, label_1 = self.data[index]
        label_2 = label_1
        if random.random() > 0.5:
            _, text, label_2 = self.data[random.randint(0, len(self.data)-1)]
        return img, text, (label_1, label_2)


class SentenseDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        data = functools.reduce(lambda a, b: a + b, data)
        self.args = args
        self.data = []
        for i, item in enumerate(data):
            tokens = tokenizer.encode(item["caption"])
            if len(tokens) > 77:
                item["caption"] = tokenizer.decode(tokens[1:75])
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        engs = []
        thas = []
        labels = []
        for sample in batch:
            eng, tha = sample.texts
            engs.append(eng)
            thas.append(tha)
            labels.append(sample.label)
        return engs, thas, torch.tensor(labels)

    def __getitem__(self, index):
        item = self.data[index]  # img, text, label_1
        text = item["caption"]
        label_1 = item["id"]
        img = Image.open(os.path.join(self.args.image_root_path, item["file_path"]))
        label_2 = label_1
        if random.random() > 0.5:
            item2 = self.data[random.randint(0, len(self.data)-1)]
            label_2 = item2["id"]
            text = item2["caption"]
        label = 0.8 if label_2 == label_1 else 0.3
        return InputExample(texts=[img, text], label=label)


class ClipDataset(Dataset):
    def __init__(self, args, data, tokenizer, transform, max_len):
        self.data = functools.reduce(lambda a, b: a + b, data)
        self.transform = transform
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]  # img, text, label_1
        text = item["caption"]
        text = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        img = Image.open(os.path.join(self.args.image_root_path, item["file_path"]))
        img = self.transform(img)
        return img, text, item["id"]


class CMPMDataset(SiameseDataset):
    def __getitem__(self, index):
        img, text, label_1 = self.data[index]
        return img, text, label_1


class TripLetDataset(SiameseDataset):
    def __getitem__(self, index):
        img, text, label_1 = self.data[index]
        neg = self.data.data[self.data.data['label'] != label_1]
        row = neg.sample(1).iloc[0]
        if row['text'] not in self.data.features:
            txt_vector = torch.load(os.path.join(self.data.folder, self.data.stage, row['file']), map_location='cpu')
            self.data.features[row['text']] = txt_vector
        return img, text, self.data.features[row['text']]
