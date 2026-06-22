from PIL import Image
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.datautils import AugMixAugmenter


def build_eval_transform(resolution: int, normalize):
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_multiview_transform(resolution: int, normalize, num_views: int, augmix: bool):
    base_transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=BICUBIC),
            transforms.CenterCrop(resolution),
        ]
    )
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    return AugMixAugmenter(
        base_transform=base_transform,
        preprocess=preprocess,
        n_views=max(0, int(num_views) - 1),
        augmix=augmix,
    )
