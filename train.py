from models.matching import ImageModel
from datagen import JSONDataset, split, TripLetDataset, SiameseDataset
import argparse
from sentence_transformers import SentenceTransformer
import torch
from utils import GradualWarmupScheduler, test_map
import tensorflow as tf
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        output1, output2 = output
        label = label.float()
        euclidean_distance = torch.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
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
    parser.add_argument("--text_vector_path", type=str, default="/media/palm/Data/tipcb/text_vectors")
    parser.add_argument("--image_root_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs")
    parser.add_argument("--json_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all.json")
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--regen', type=bool, default=False)

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
    train_dataset = JSONDataset(args, train_list, text_model, regen=args.regen, train=True)  # args, data, regen, train
    val_dataset = JSONDataset(args, val_list, text_model, regen=args.regen, train=False)

    train_dataset = TripLetDataset(train_dataset)  # todo: refactory nightmare
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
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    train_steps = 0
    test_steps = 0
    writer = SummaryWriter(log_dir=f'{args.log_dir}/{folder}')
    for epoch in range(args.num_epochs):
        print('Epoch:', epoch + 1)
        progbar = tf.keras.utils.Progbar(len(train_loader))
        image_model.train()
        for idx, (image, positive, negative) in enumerate(train_loader):
            image = image.to(args.device)
            positive = positive.to(args.device)
            negative = negative.to(args.device)
            image_feature = image_model(image)
            loss = criterion(image_feature, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [('loss', loss.cpu().detach().numpy()),
                        ]
            writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), train_steps)
            train_steps += 1
            progbar.update(idx + 1, printlog)
        scheduler_warmup.step()
        progbar = tf.keras.utils.Progbar(len(val_loader))
        query_feature = torch.empty((0, text_model[1].pooling_output_dimension), dtype=torch.float32)
        query_label = torch.empty((0, ), dtype=torch.int)
        gallery_feature = torch.empty((0, text_model[1].pooling_output_dimension), dtype=torch.float32)
        gallery_label = torch.empty((0, ), dtype=torch.int)
        with torch.no_grad():
            image_model.eval()
            for idx, (image, text, labels) in enumerate(val_loader):
                image = image.to(args.device)
                text = text.to(args.device)
                image_feature = image_model(image)
                loss = criterion([text, image_feature], (labels[0] == labels[1]).view(-1).to(args.device).float())
                printlog = [('val_loss', loss.cpu().detach().numpy()),
                            ]
                progbar.update(idx + 1, printlog)
                query_feature = torch.cat((query_feature, text.cpu()))
                gallery_feature = torch.cat((gallery_feature, image_feature.cpu()))
                query_label = torch.cat((query_label, labels[1]))
                gallery_label = torch.cat((gallery_label, labels[0]))
                writer.add_scalar('Loss/test', loss.cpu().detach().numpy(), test_steps)
                test_steps += 1
        cm0, cm4, cm9, ap = test_map(query_feature, query_label, gallery_feature, gallery_label)
        writer.add_scalar('Metrics/r@1', cm0, test_steps)
        writer.add_scalar('Metrics/r@5', cm4, test_steps)
        writer.add_scalar('Metrics/r@10', cm9, test_steps)
        writer.add_scalar('Metrics/ap', ap, test_steps)
