from models.matching import ImageModel
from datagen import JSONDataset, split, TripLetDataset, SiameseDataset
import argparse
from sentence_transformers import SentenceTransformer
import torch
from utils import test_map
import tensorflow as tf
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
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


class CMPMLoss(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, image_embeddings, text_embeddings, labels):
        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon)) # (4)
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return cmpm_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.0004)
    parser.add_argument('--warm_lr_init', type=float, default=1e-6)
    parser.add_argument('--warm_epoch', default=10, type=int)

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
    schedule = CosineLRScheduler(optimizer,
                                 t_initial=args.num_epochs,
                                 cycle_mul=1,
                                 lr_min=1e-8,
                                 cycle_decay=0.1,
                                 warmup_lr_init=args.warm_lr_init,
                                 warmup_t=args.warm_epoch,
                                 cycle_limit=1,
                                 t_in_epochs=True,
                                 noise_range_t=None,
                                 )
    mse = nn.MSELoss()
    identity = nn.Identity()
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    contrastive = ContrastiveLoss()
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
        # writer.add_scalar('OPtimizer/LR', schedule._get_lr(epoch+1), epoch)
        schedule.step(epoch + 1)
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
                # todo: distance here should be labels[0] != labels[1]
                loss = contrastive([text, image_feature], (labels[0] != labels[1]).view(-1).to(args.device).float())
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
        writer.add_scalar('Metrics/r@1', cm0, epoch)
        writer.add_scalar('Metrics/r@5', cm4, epoch)
        writer.add_scalar('Metrics/r@10', cm9, epoch)
        writer.add_scalar('Metrics/ap', ap, epoch)
