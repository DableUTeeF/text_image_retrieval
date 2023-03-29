from sentence_transformers import SentenceTransformer, util
from PIL import Image

if __name__ == '__main__':
    model = SentenceTransformer('/media/palm/Data/tipcb/checkpoint')

    img_emb = model.encode(Image.open('/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs/test_query/p10376_s14337.jpg'))

    text_emb = model.encode([
        "A woman walking next to an animal wearing a full length dress and dark coloured shoes.",
        "一个女人走在一个动物旁边，穿着长裙和深色的鞋子。",
        "一个穿着蓝色衬衫、一条蓝色短裤和一双灰色鞋子的女人。",
        "A man wearing a white shirt, a pair of black pants and a pair of shoes.",

    ])

    cos_scores = util.cos_sim(img_emb, text_emb)
    print(cos_scores)
