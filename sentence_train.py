from sentence_transformers import SentenceTransformer, losses, evaluation
from torch.utils.data import DataLoader
from datagen import SentenseDataset, JSONDataset, split
import argparse
from utils import CLIPSimilarityEvaluator

if __name__ == '__main__':
    # Define the model. Either from scratch of by loading a pre-trained model
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--output_path', type=str, default='/media/palm/Data/tipcb/checkpoint')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.0004)
    parser.add_argument('--warm_lr_init', type=float, default=1e-6)
    parser.add_argument('--warm_epoch', default=10, type=int)

    # models
    parser.add_argument('--model', type=str, default='clip-ViT-B-32')

    # dataset
    parser.add_argument("--text_vector_path", type=str, default="/media/palm/Data/tipcb/text_vectors")
    parser.add_argument("--image_root_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs")
    parser.add_argument("--json_path", type=str, default="/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all.json")
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--regen', type=bool, default=False)

    args = parser.parse_args()
    model = SentenceTransformer(args.model)
    tokenizer = model._first_module().processor.tokenizer

    train_list, val_list = split(args)

    train_dataset = SentenseDataset(args, train_list, tokenizer)
    val_dataset = SentenseDataset(args, val_list, tokenizer)


    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=SentenseDataset.collate_fn)
    train_loss = losses.CosineSimilarityLoss(model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_dataset)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100,
              evaluator=evaluator,
              output_path=args.output_path)
