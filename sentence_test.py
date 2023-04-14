from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from datagen import split
import argparse
import functools
import os
from utils import test_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_path', type=str, default='/media/palm/Data/tipcb/checkpoint')
    parser.add_argument("--image_root_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs")
    parser.add_argument("--json_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all.json")
    parser.add_argument('--dim', type=int, default=512)

    args = parser.parse_args()
    model = SentenceTransformer('/media/palm/Data/tipcb/checkpoint/remove77')
    tokenizer = model._first_module().processor.tokenizer

    train_list, val_list = split(args)
    val_list = functools.reduce(lambda a, b: a + b, val_list)

    query_feature = torch.empty((0, args.dim), dtype=torch.float32)
    query_label = torch.empty((0,), dtype=torch.int)
    gallery_feature = torch.empty((0, args.dim), dtype=torch.float32)
    gallery_label = torch.empty((0,), dtype=torch.int)

    for i, item in enumerate(val_list):
        tokens = tokenizer.encode(item["caption"])
        if len(tokens) > 77:
            continue
            item["caption"] = tokenizer.decode(tokens[1:75])
        text = item["caption"]
        text_emb = model.encode([text], convert_to_tensor=True)
        img = Image.open(os.path.join(args.image_root_path, item["file_path"]))
        img_emb = model.encode([img], convert_to_tensor=True)
        label = torch.tensor([item["id"]])
        query_feature = torch.cat((query_feature, text_emb.cpu()))
        gallery_feature = torch.cat((gallery_feature, img_emb.cpu()))
        query_label = torch.cat((query_label, label))
        gallery_label = torch.cat((gallery_label, label))
    cm0, cm4, cm9, ap = test_map(query_feature, query_label, gallery_feature, gallery_label)
    print(f'R@1: {cm0}, R@5: {cm4}, R@10: {cm9}, MAP: {ap}')

