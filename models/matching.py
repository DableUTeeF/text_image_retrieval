from torch import nn
from models.focalnet import focalnet_tiny_srf
import torch


class ImageModel(nn.Module):
    def __init__(self, txt_size, w=128, h=384, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size or txt_size
        self.backbone = focalnet_tiny_srf(pretrained=True)
        self.image_features = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, txt_size)
        )


    def forward(self, img):
        x, H, W = self.backbone.patch_embed(img)
        x = self.backbone.pos_drop(x)

        for layer in self.backbone.layers:
            x, H, W = layer(x, H, W)
        x = self.backbone.norm(x)  # B L C
        x = self.backbone.avgpool(x.transpose(1, 2))  # B C 1
        x = self.image_features(x.view(x.size(0), -1))
        return x


if __name__ == '__main__':
    model = FeatureMatching(0, 0)
    model(None, torch.zeros((2, 3, 384, 128)))
