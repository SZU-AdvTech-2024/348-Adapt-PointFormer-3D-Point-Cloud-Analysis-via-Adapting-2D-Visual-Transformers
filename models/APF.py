import os

import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import timm
from matplotlib import pyplot as plt

from models.point_pn import Point_PN_scan, Point_PN_mn40, ViT
from models.transformer import *
import torch.nn.functional as F
from plotting import plotting
from plotting2 import plotting2

class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=30):
        super(CosineClassifier, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=0)
        return F.linear(x, weight) * self.scale

class APF(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_large = Point_PN_mn40()
        self.vision = ViT()
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.num_classes)
        )
        # self.head = CosineClassifier(768, 40)
        self.dropout = nn.Dropout(0.1)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        for name, param in self.named_parameters():
            if 'head' in name or 'enc_norm' in name or 'encoder' in name or 'conv_block' in name\
                    or 'pos_embed' in name:
                param.requires_grad = True
                print(name, param.shape)

    def forward(self, x, xyz):#[32,6,8192], [32.8192,3]
        x, prompt, xyz2 = self.encoder_large(x, xyz)
        new_x = x.max(1)[0] + x.mean(1)
        x, f1, f2, f3 = self.vision(x, prompt)
        x = self.dropout(x)
        # x = x + new_x
        x = self.head(x)
        # plotting(xyz, xyz2)
        # plotting2(xyz, xyz2)
        return x

def create_point_former(num_classes):
    vit = timm. create_model("vit_base_patch16_224_in21k", pretrained=True)
    checkpoint = torch.load(r'/root/wish/Point-NN-main/ViT-L-14.pt')
    checks = torch.load(r'/root/wish/Point-PN-main/imagebind_audio.pth')
    audio = {}
    for k, v in checks.items():
        if 'visual' not in k:
            new_k = k.replace("modality_trunks.audio", "vision")
            new_k = new_k.replace("in_proj_", "qkv.")
            new_k = new_k.replace("out_", "")
            new_k = new_k.replace("_1", "1")
            new_k = new_k.replace("_2", "2")
            audio[new_k] = v
    checkpoint = checkpoint.state_dict()
    clip_text = {}
    for k, v in checkpoint.items():
        if 'visual' not in k:
            new_k = k.replace("transformer.resblocks", "vision.blocks")
            new_k = new_k.replace("ln_", "norm")
            new_k = new_k.replace("c_fc", "fc1")
            new_k = new_k.replace("c_proj", "fc2")
            new_k = new_k.replace("proj_", "qkv.")
            new_k = new_k.replace("in_", "")
            new_k = new_k.replace("out_", "")
            new_k = new_k.replace("layer_", "")
            clip_text[new_k] = v
    base_state_dict = vit.state_dict()
    vit_dict = {}
    for k, v in base_state_dict.items():
        if 'block' in k:
            new_k = k.replace("blocks", "vision.blocks")
            vit_dict[new_k] = v
        elif 'cls_token' in k:
            new_k = k.replace("cls_token", "vision.cls_token")
            vit_dict[new_k] = v
        else:
            vit_dict[k] = v
    del vit_dict['head.weight']
    del vit_dict['head.bias']
    model = APF(num_classes=num_classes)
    print('--------------------------------------------------------------------------')
    model.load_state_dict(base_state_dict, False)
    model.freeze()
    return model


if __name__ == '__main__':
    x = torch.rand(32, 1024, 6).to('cuda')
    model = create_point_former(num_classes=40).to('cuda')
    x = model(x.permute(0, 2, 1), x[:, :, :3])
    tuning_num_params = 0
    num_params = 0
    a = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            tuning_num_params += p.numel()
        else:
            a += p.numel()
    print("===============================================")
    print("model parameters: " + str(num_params))
    print("model tuning parameters: " + str(tuning_num_params))
    print("model not tuning parameters: " + str(a))
    print("tuning rate: " + str(tuning_num_params / num_params))
    print("===============================================")
    # for name, param in model.named_parameters():
        # if param.requires_grad is True:
        #     print(f"{name},{param.shape}")
    print(x.shape)



