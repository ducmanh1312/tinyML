import torch

from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce



def img_process(img):
    transform = Compose([Resize(224,224), ToTensor()])
    x = transform(img)
    x = x.unsqueeze(0)

    return x


class PatchEmbedding(nn.Module):
    def __init__(self, channels=3, patch_size=16, embed_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()

        self.LinearProjection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * channels, embed_size))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.position = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embed_size))

    def forward(self, x):
        b, _, _, _ = x.shape  # b c h w
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)

        x = self.LinearProjection(x)
        x = torch.cat([cls_token, x], dim=1)

        x += x + self.position

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size=768, dropout_p=0, expansion=4):
        super(TransformerEncoderBlock, self).__init__()

        self.key = nn.Linear(embed_size, embed_size)
        self.query = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.LN = nn.LayerNorm(embed_size)
        self.attn = nn.MultiheadAttention(embed_size, 12, dropout_p) # default head_count = 12

        self.layer2 = nn.Sequential(nn.LayerNorm(embed_size),
                                    nn.Linear(embed_size, embed_size * expansion),
                                    nn.GELU(),
                                    nn.Dropout(),
                                    nn.Linear(embed_size * expansion, embed_size))

    def forward(self, x):
        feature_map1 = self.layer1(x)

        map2 = self.layer2(feature_map1)
        feature_map2 = map2 + feature_map1

        return feature_map2

    def layer1(self, x):
        LN_map = self.LN(x)
        k = self.key(LN_map)
        q = self.query(LN_map)
        v = self.value(LN_map)

        attn_map, _ = self.attn(q, k, v)
        return attn_map + x


class TransformerEncoder(nn.Module):
    def __init__(self, layer_num=6, embed_size=768, dropout_p=0, expansion=4):
        super(TransformerEncoder, self).__init__()

        self.encoder_layers = self.make_layer(layer_num, embed_size, dropout_p, expansion)

    def forward(self, x):
        return self.encoder_layers(x)

    def make_layer(self, layer_num, embed_size, dropout_p, expansion):
        layer = []
        for i in range(layer_num):
            layer.append(TransformerEncoderBlock(embed_size, dropout_p, expansion))

        return nn.Sequential(*layer)


class ClassificationHead(nn.Module):
    def __init__(self, embed_size=768, class_num=10):
        super(ClassificationHead, self).__init__()

        self.head = nn.Sequential(Reduce('b n e->b e', reduction='mean'),
                                  nn.LayerNorm(embed_size),
                                  nn.Linear(embed_size, class_num))

    def forward(self, x):
        return self.head(x)


class VisionTransformer(nn.Module):
    def __init__(self, channels=3, patch_size=16, embed_size=768, img_size=224, block_num=12, class_num=118):
        super(VisionTransformer, self).__init__()

        self.process = nn.Sequential(PatchEmbedding(channels, patch_size, embed_size, img_size),
                                     TransformerEncoder(block_num, embed_size),
                                     ClassificationHead(embed_size, 1024))
#        self.process = nn.Sequential(PatchEmbedding(channels, patch_size, embed_size, img_size),
#                                     TransformerEncoder(block_num, embed_size),
#                                     )
                                     
    def forward(self, x):
        #print(self.process(x).shape)
        return self.process(x)
