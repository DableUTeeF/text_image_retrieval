from clip_train import Args
from models.clip import VisionEncoder, TextEncoder, clip_loss, metrics
import torch
from datagen import split, build_transform
from PIL import Image
import os
from transformers import AutoTokenizer

def caption_encode(caption, caption_encoder):
    out = caption_encoder.base(
        input_ids=caption['input_ids'].to(device),
        token_type_ids=caption['token_type_ids'].to(device),
        attention_mask=caption['attention_mask'].to(device),
    )[0]
    out = out[:, 0, :]
    projected_vec = caption_encoder.projection(out)
    projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
    return projected_vec / projection_len


if __name__ == '__main__':
    args = Args()
    device = 'cuda'
    vision_encoder = VisionEncoder(args.embed_dim).to(device).eval()
    caption_encoder = TextEncoder(args.text_model, args.transformer_embed_dim, args.embed_dim).to(device).eval()
    checkpoint = torch.load('/media/palm/BiggerData/caption/cp/clip/044.pth', map_location='cpu')
    vision_encoder.load_state_dict(checkpoint['vision_encoder'])
    caption_encoder.load_state_dict(checkpoint['caption_encoder'])
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    train_list, val_list = split(args)
    test_transform = build_transform(False, args)

    im1 = test_transform(Image.open(os.path.join(args.image_root_path, val_list[0][0]['file_path']))).to(device)
    im2 = test_transform(Image.open(os.path.join(args.image_root_path, val_list[1][0]['file_path']))).to(device)
    caption1 = tokenizer(val_list[0][0]['caption'], max_length=args.max_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt")
    caption2 = tokenizer(val_list[1][0]['caption'], max_length=args.max_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt")
    with torch.no_grad():
        image_embed1 = vision_encoder(im1.unsqueeze(0))
        image_embed2 = vision_encoder(im2.unsqueeze(0))
        caption_embed1 = caption_encode(caption1, caption_encoder)
        caption_embed2 = caption_encode(caption2, caption_encoder)
    print()
