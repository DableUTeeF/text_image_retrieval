from models.matching import ImageModel
from datagen import JSONDataset, split, SiameseDataset
import argparse
from sentence_transformers import SentenceTransformer
import torch
from utils import GradualWarmupScheduler, test_map
import tensorflow as tf
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')

    # optimizer
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--wd', type=float, default=0.00004)
    parser.add_argument('--warm_epoch', default=10, type=int)
    parser.add_argument('--epoches_decay', type=int, default=40)  # todo: sounds like a dumb idea
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)

    # models
    parser.add_argument('--text_model', type=str, default='all-MiniLM-L6-v2')
    # parser.add_argument('--img_model', type=str, default='focalnet_tiny_srf')

    # dataset
    parser.add_argument("--image_root_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs")
    parser.add_argument("--json_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all.json")
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=384)

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    folder = f'log_{len(os.listdir(args.log_dir))}'
    os.mkdir(os.path.join(args.log_dir, folder))

    text_model = SentenceTransformer(args.text_model)
    text_model.eval()
    text_model.to(args.device)
    image_model = ImageModel(text_model[1].pooling_output_dimension)
    image_model.to(args.device)

    train_list, val_list = split(args)

    train_dataset = JSONDataset(args, train_list, train=True)  # args, data, regen, train
    val_dataset = JSONDataset(args, val_list, train=False)

    train_dataset = SiameseDataset(train_dataset)  # todo: refactory nightmare
    val_dataset = SiameseDataset(val_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4
                            )

    optimizer = torch.optim.AdamW(image_model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay), gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer,
                                              multiplier=1,
                                              total_epoch=args.warm_epoch,
                                              after_scheduler=scheduler_steplr)
    mse = nn.MSELoss()
    identity = nn.Identity()
    train_log = {  # todo: really need to improve logging
        'step': [],
        'epoch': [],
        'similarity': [],
        'mse': []
    }
    test_log = {
        'step': [],
        'epoch': [],
        'similarity': [],
        'mse': []
    }
    on_epoch_log = {
        'epoch': [],
        'ap': [],
        'cm0': [],
        'cm4': [],
        'cm9': [],
    }
    train_steps = 0
    test_steps = 0
    for epoch in range(args.num_epochs):
        print('Epoch:', epoch + 1)
        progbar = tf.keras.utils.Progbar(len(train_loader))
        image_model.train()
        for idx, (image, text, labels) in enumerate(train_loader):
            image = image.to(args.device)
            with torch.no_grad():
                text_features = text_model.encode(text, convert_to_tensor=True)
            image_feature = image_model(image)
            similarity = identity(torch.cosine_similarity(text_features, image_feature))
            loss = mse(similarity, (labels[0] == labels[1]).view(-1).to(args.device).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [('similarity', similarity.cpu().detach().numpy()),
                        ('mse', loss.cpu().detach().numpy()),
                        ]
            train_log['step'].append(train_steps)
            train_log['epoch'].append(epoch)
            train_log['similarity'].append(similarity.cpu().detach().numpy())
            train_log['mse'].append(loss.cpu().detach().numpy())
            train_steps += 1
            progbar.update(idx + 1, printlog)
        pd.DataFrame(train_log).to_csv(f'{args.log_dir}/{folder}/train.csv', index=False)
        scheduler_warmup.step()
        progbar = tf.keras.utils.Progbar(len(val_loader))
        query_feature = []
        query_label = []
        gallery_feature = []
        gallery_label = []
        with torch.no_grad():
            image_model.eval()
            for idx, (image, text, labels) in enumerate(val_loader):
                labels = labels.to(args.device)
                image = image.to(args.device)
                text_features = text_model.encode(text, convert_to_tensor=True)
                image_feature = image_model(image)
                similarity = identity(torch.cosine_similarity(text_features, image_feature))
                loss = mse(similarity, (labels[0] == labels[1]).view(-1).to(args.device).float())
                printlog = [('val_similarity', similarity.cpu().detach().numpy()),
                            ('val_mse', loss.cpu().detach().numpy()),
                            ]
                query_feature.append(text_features.cpu())
                gallery_feature.append(image_feature.cpu())
                query_label.append(labels[1])
                gallery_label.append(labels[0])
                progbar.update(idx + 1, printlog)
                test_log['step'].append(test_steps)
                test_log['epoch'].append(epoch)
                test_log['similarity'].append(similarity.cpu().detach().numpy())
                test_log['mse'].append(loss.cpu().detach().numpy())
                test_steps += 1
        cm0, cm4, cm9, ap = test_map(query_feature, query_label, gallery_feature, gallery_label)
        on_epoch_log['epoch'].append(epoch)
        on_epoch_log['cm4'].append(cm4)
        on_epoch_log['cm0'].append(cm0)
        on_epoch_log['cm9'].append(cm9)
        on_epoch_log['ap'].append(ap)
        pd.DataFrame(test_log).to_csv(f'{args.log_dir}/{folder}/test.csv', index=False)
        pd.DataFrame(on_epoch_log).to_csv(f'{args.log_dir}/{folder}/epochs.csv', index=False)
