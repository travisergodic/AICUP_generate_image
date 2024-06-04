import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, ResNet34_Weights, vgg16, resnet34

from .inception import fid_inception_v3


def build_vgg16():
    model = vgg16(VGG16_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    return model


def build_resnet34():
    model = resnet34(ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    return model


def build_clip_vit_b32():
    import clip
    model, _ = clip.load('ViT-B/32', "cpu")
    return model.visual


def build_clip_vit_b16():
    import clip
    model, _ = clip.load('ViT-B/16', "cpu")
    return model.visual

def build_clip_vit_l14():
    import clip
    model, _ = clip.load('ViT-L/14', "cpu")
    return model.visual


def build_dinov2_vit_b14():
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

def build_inceptionv3():
    return fid_inception_v3()