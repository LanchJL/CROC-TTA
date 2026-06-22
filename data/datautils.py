import os
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME={
    'I': 'imagenet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2',
    'flower102': 'Flower102',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}


class ReorderableDataset(torch.utils.data.Dataset):
    """Dataset wrapper with deterministic subset reordering support."""

    def __init__(self, dataset):
        self.dataset = dataset
        self._subset_indices = np.arange(len(dataset), dtype=np.int64)

    def set_specific_subset(self, indices):
        arr = np.asarray(indices, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("indices must be a 1D list or array")
        if arr.size > 0 and (arr.min() < 0 or arr.max() >= len(self.dataset)):
            raise IndexError("subset index out of range")
        self._subset_indices = arr

    def reset_specific_subset(self):
        self._subset_indices = np.arange(len(self.dataset), dtype=np.int64)

    @property
    def targets(self):
        if hasattr(self.dataset, "targets"):
            base = np.asarray(self.dataset.targets)
            return base[self._subset_indices].tolist()
        return None

    @property
    def samples(self):
        if hasattr(self.dataset, "samples"):
            return [self.dataset.samples[int(i)] for i in self._subset_indices]
        return None

    def __len__(self):
        return int(self._subset_indices.shape[0])

    def __getitem__(self, idx):
        base_idx = int(self._subset_indices[int(idx)])
        return self.dataset[base_idx]

    def __getattr__(self, name):
        return getattr(self.dataset, name)

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'images')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id.lower() in {d.lower() for d in fewshot_datasets}:
        set_id = set_id.lower()
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    else:
        raise NotImplementedError
        
    return ReorderableDataset(testset)


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views
