from sentence_transformers import SentenceTransformer, losses, evaluation
from torch.utils.data import DataLoader
from datagen import SentenseDataset, JSONDataset, split
import argparse
from PIL import Image
import os
import torch
from utils import test_map
from torch.utils.tensorboard import SummaryWriter


class TIPCBEvaluator(evaluation.SentenceEvaluator):
    def __init__(self, val_list, writer):
        self.val_list = val_list
        self.writer = writer

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        query_feature = torch.empty((0, args.embed_dim), dtype=torch.float32)
        query_label = torch.empty((0,), dtype=torch.int)
        gallery_feature = torch.empty((0, args.embed_dim), dtype=torch.float32)
        gallery_label = torch.empty((0,), dtype=torch.int)
        for i, item in enumerate(self.val_list):
            tokens = tokenizer.encode(item["caption"])
            if len(tokens) > 77:
                continue
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
        self.writer.add_scalar('Metrics/r@1', cm0, epoch)
        self.writer.add_scalar('Metrics/r@5', cm4, epoch)
        self.writer.add_scalar('Metrics/r@10', cm9, epoch)
        self.writer.add_scalar('Metrics/ap', ap, epoch)
        return ap


if __name__ == '__main__':
    # Define the model. Either from scratch of by loading a pre-trained model
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=45)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--output_path', type=str, default='/media/palm/Data/tipcb/checkpoint/remove77_2')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.0004)
    parser.add_argument('--warm_lr_init', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', default=200, type=int)

    # models
    parser.add_argument('--model', type=str, default='clip-ViT-B-32')

    # dataset
    parser.add_argument("--text_vector_path", type=str, default="/media/palm/Data/tipcb/text_vectors")
    parser.add_argument("--image_root_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs")
    parser.add_argument("--json_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all_encn.json")
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--regen', type=bool, default=False)

    args = parser.parse_args()
    model = SentenceTransformer(args.model)
    tokenizer = model._first_module().processor.tokenizer

    train_list, val_list = split(args)

    train_dataset = SentenseDataset(args, train_list, tokenizer)
    # val_dataset = SentenseDataset(args, val_list, tokenizer)


    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=SentenseDataset.collate_fn)
    train_loss = losses.CosineSimilarityLoss(model)
    # evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_dataset)
    writer = SummaryWriter(log_dir=f'{args.output_path}/log')
    evaluator = TIPCBEvaluator(val_list, writer)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=args.num_epochs,
              warmup_steps=args.warmup_steps, evaluator=evaluator,
              output_path=args.output_path)
