from models.clip import VisionEncoder, TextEncoder, clip_loss, metrics, ClipLoss
from transformers import AutoTokenizer
from datagen import ClipDataset, split, build_transform
from torch.utils.data import DataLoader
from torch import optim
from timm.scheduler import CosineLRScheduler
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import test_map


class Args:
    embed_dim = 512
    transformer_embed_dim = 768
    max_len = 128
    text_model = "bert-base-uncased"

    epochs = 45
    warmup_epochs = 3
    batch_size = 16
    wd = 0.05
    base_lr = 5e-4
    warmup_lr = 5e-7
    min_lr = 5e-6
    width = 128
    height = 384

    device = 'cuda'

    json_path = '/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all.json'
    output_path = '/media/palm/Data/tipcb/checkpoint/clip/'
    image_root_path = '/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs'


def common_step(images, text):
    images = images.to(args.device)
    text = text.to(args.device)

    image_embed = vision_encoder(images)
    caption_embed = caption_encoder(text)
    similarity = caption_embed @ image_embed.T

    loss = criterion(image_embed, caption_embed, 1)
    img_acc, cap_acc = metrics(similarity)
    return loss, img_acc, cap_acc, image_embed, caption_embed


if __name__ == '__main__':
    args = Args()

    vision_encoder = VisionEncoder(args.embed_dim).to(args.device)
    caption_encoder = TextEncoder(args.text_model, args.transformer_embed_dim, args.embed_dim).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    criterion = ClipLoss().to(args.device)

    linear_scaled_lr = args.base_lr #* args.batch_size / 512.0
    linear_scaled_warmup_lr = args.warmup_lr #* args.batch_size / 512.0
    linear_scaled_min_lr = args.min_lr #* args.batch_size / 512.0
    optimizer = optim.AdamW(
        [
            {'params': vision_encoder.parameters(), 'lr': linear_scaled_lr},
            {'params': caption_encoder.parameters(), 'lr': linear_scaled_lr * 0.1}
        ],
        weight_decay=args.wd
    )

    train_list, val_list = split(args)
    train_dataset = ClipDataset(args, train_list, tokenizer, transform=build_transform(True, args), max_len=args.max_len)
    val_dataset = ClipDataset(args, val_list, tokenizer, transform=build_transform(False, args), max_len=args.max_len)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=2)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=2)

    num_steps = int(args.epochs * len(train_dataloader))
    warmup_steps = int(args.warmup_epochs * len(train_dataloader))
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        cycle_mul=1.,
        lr_min=linear_scaled_min_lr,
        warmup_lr_init=linear_scaled_warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    writer = SummaryWriter(log_dir=f'{args.output_path}/log')
    max_acc = 0
    for epoch in range(args.epochs):
        img_accs = []
        cap_accs = []
        losses = []
        for idx, (images, text, _) in enumerate(train_dataloader):
            for key in text:
                text[key] = text[key][:, 0]
            loss, img_acc, cap_acc, _, _ = common_step(images, text)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step_update(epoch * len(train_dataloader) + idx)
            losses.append(loss.cpu().detach().numpy())
            img_accs.append(img_acc.cpu().detach().numpy())
            cap_accs.append(cap_acc.cpu().detach().numpy())
            writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), epoch * len(train_dataloader) + idx)
            writer.add_scalar('img_accs/train', img_acc.cpu().detach().numpy(), epoch * len(train_dataloader) + idx)
            writer.add_scalar('cap_accs/train', cap_acc.cpu().detach().numpy(), epoch * len(train_dataloader) + idx)
            if idx % 100 == 0:
                print(
                    f'epoch: {epoch}/{args.epochs} - step: {idx}/{len(train_dataloader)} - loss: {np.mean(losses):.4f} - img_acc: {np.mean(img_accs):.4f} - cap_acc: {np.mean(cap_accs):.4f}')
                img_accs = []
                cap_accs = []
                losses = []

        with torch.no_grad():
            img_accs = []
            cap_accs = []
            losses = []
            query_feature = torch.empty((0, args.embed_dim), dtype=torch.float32)
            query_label = torch.empty((0,), dtype=torch.int)
            gallery_feature = torch.empty((0, args.embed_dim), dtype=torch.float32)
            gallery_label = torch.empty((0,), dtype=torch.int)
            for idx, (images, text, label) in enumerate(val_dataloader):
                for key in text:
                    text[key] = text[key][:, 0]
                loss, img_acc, cap_acc, image_embed, caption_embed = common_step(images, text)
                losses.append(loss.cpu().detach().numpy())
                img_accs.append(img_acc.cpu().detach().numpy())
                cap_accs.append(cap_acc.cpu().detach().numpy())
                query_feature = torch.cat((query_feature, caption_embed.cpu()))
                gallery_feature = torch.cat((gallery_feature, image_embed.cpu()))
                query_label = torch.cat((query_label, label))
                gallery_label = torch.cat((gallery_label, label))
                writer.add_scalar('Loss/val', loss.cpu().detach().numpy(), epoch * len(val_dataloader) + idx)
                writer.add_scalar('img_accs/val', img_acc.cpu().detach().numpy(), epoch * len(val_dataloader) + idx)
                writer.add_scalar('cap_accs/val', cap_acc.cpu().detach().numpy(), epoch * len(val_dataloader) + idx)
            cm0, cm4, cm9, ap = test_map(query_feature, query_label, gallery_feature, gallery_label)
            writer.add_scalar('Metrics/r@1', cm0, epoch)
            writer.add_scalar('Metrics/r@5', cm4, epoch)
            writer.add_scalar('Metrics/r@10', cm9, epoch)
            writer.add_scalar('Metrics/ap', ap, epoch)
            writer.add_scalar('Metrics/img_acc', np.mean(img_accs), epoch)
            writer.add_scalar('Metrics/cap_acc', np.mean(cap_accs), epoch)

            print(f'val_epoch: {epoch} - loss: {np.mean(losses):.4f} - img_acc: {np.mean(img_accs):.4f} - cap_acc: {np.mean(cap_accs):.4f} - rank1: {cm0:.4f} - map: {ap:.4f}')
            torch.save({
                'vision_encoder': vision_encoder.state_dict(),
                'caption_encoder': caption_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'{args.output_path}/last.pth')
            if np.mean(cap_accs) + np.mean(img_accs) > max_acc:
                max_acc = np.mean(cap_accs) + np.mean(img_accs)
                torch.save({
                    'vision_encoder': vision_encoder.state_dict(),
                    'caption_encoder': caption_encoder.state_dict(),
                }, f'{args.output_path}/best.pth')
