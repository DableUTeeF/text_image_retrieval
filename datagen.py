from torch.utils.data import Dataset
import torchvision.transforms as transforms
import functools
import os
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import random


def split(args):
    with open(args.json_path, "rb") as f:
        caption_all = json.load(f)

    # group all the record (each element of `caption_all`) by id
    group_by_id = dict()
    for record in caption_all:
        # check if image file doesn't exist
        if not os.path.exists(os.path.join(args.image_root_path, record["file_path"])):
            continue
        # if record["file_path"].split("/")[0] not in ["test_query", "train_query"]:
        #     continue
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
    def __init__(self, args, data, train):
        self.data = functools.reduce(lambda a, b: a + b, data)  # todo: add class count for triplet loss
        self.train = train
        self.args = args
        self.stage = 'train' if train else 'val'

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

    def __getitem__(self, index):
        item = self.data[index]
        text = item["caption"]
        label = item["id"]
        img = Image.open(os.path.join(self.args.image_root_path, item["file_path"]))
        img = self.transform(img)
        return img, text, label

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
        if random.random() > 0.5:
            _, text, label_2 = self.data[random.randint(0, len(self.data))]
        return img, text, (label_1, label_2)

