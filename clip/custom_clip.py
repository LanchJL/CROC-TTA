import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load
from model.embed_with_prompts import build_text_features


DOWNLOAD_ROOT = "~/.cache/clip"


class FrozenCLIPClassifier(nn.Module):
    def __init__(self, args, device, classnames, dataset, arch="ViT-B/16"):
        super().__init__()
        clip_model, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)

        self.clip = clip_model
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.detach()
        self.args = args
        self.device = device
        self.dataset = dataset
        self.classnames = []
        self.text_features = None
        self.reset_classnames(args, classnames, dataset)

    @property
    def dtype(self):
        return self.clip.dtype

    def reset_classnames(self, args, classnames, dataset):
        self.args = args
        self.dataset = dataset
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.text_features = None

    def init_text_features(self):
        text_features = build_text_features(
            self.args,
            self.clip,
            self.dataset,
            self.classnames,
            self.device,
        )
        self.text_features = F.normalize(text_features.detach(), dim=-1)
        self.text_features.requires_grad_(False)

    @torch.no_grad()
    def encode_image(self, image):
        features = self.clip.encode_image(image.to(self.device).type(self.dtype))
        return F.normalize(features.float(), dim=-1).detach()

    @torch.no_grad()
    def encode_text(self, text):
        features = self.clip.encode_text(text.to(self.device))
        return F.normalize(features.float(), dim=-1).detach()

    @torch.no_grad()
    def forward(self, image):
        if self.text_features is None:
            raise RuntimeError("Text features are not initialized.")
        image_features = self.encode_image(image)
        return self.logit_scale.exp().float() * image_features @ self.text_features.t()


def build_frozen_clip(args, arch, dataset, device, classnames):
    return FrozenCLIPClassifier(args, device, classnames, dataset, arch=arch)
