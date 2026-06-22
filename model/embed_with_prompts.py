import json
import os
import pickle
import re

import torch
import torch.nn.functional as F

from clip import tokenize
from data.imagnet_prompts import gpt_dict, imagenet_templates, make_descriptor_sentence


def sanitize_path(input_string: str) -> str:
    return re.sub(r'[\/:*?"<>|]', "_", input_string)


def get_cache_path(args, dataset):
    dataset_key = dataset.lower()
    arch = sanitize_path(args.backbone)
    template_tag = "_templates" if args.use_templates else ""
    descriptor_tag = "_descriptors" if args.use_descriptors else ""
    name = f"{dataset_key}{template_tag}{descriptor_tag}_{arch}.pkl"
    return os.path.join(args.cache_path, name)


def _class_prompts(classname: str, use_templates: bool):
    name = classname.replace("_", " ")
    if use_templates:
        return [template.format(name) for template in imagenet_templates]
    return [f"a photo of a {name}."]


def _descriptor_prompts(dataset: str, classnames):
    data_key = "imagenet" if dataset.upper() in {"I", "A", "R", "K", "V"} else gpt_dict[dataset.lower()]
    path = os.path.join("prompts", f"{data_key}-gpt4.json")
    if not os.path.exists(path):
        return [[] for _ in classnames]
    with open(path, "r", encoding="utf-8") as handle:
        descriptors = json.load(handle)

    prompts_by_class = []
    normalized_descriptors = {key.replace("_", " ").lower(): value for key, value in descriptors.items()}
    descriptor_items = list(descriptors.values())
    use_positional_mapping = len(descriptor_items) == len(classnames)
    for class_index, classname in enumerate(classnames):
        clean_name = classname.replace("_", " ")
        class_descriptors = normalized_descriptors.get(clean_name.lower())
        if class_descriptors is None and use_positional_mapping:
            class_descriptors = descriptor_items[class_index]
        if class_descriptors is None:
            prompts_by_class.append([])
            continue
        prompts = [f"{clean_name}, {make_descriptor_sentence(desc)}" for desc in class_descriptors]
        prompts_by_class.append(prompts)
    return prompts_by_class


@torch.no_grad()
def _encode_prompt_group(clip_model, prompts, device):
    tokens = tokenize(prompts).to(device)
    features = clip_model.encode_text(tokens)
    features = F.normalize(features.float(), dim=-1)
    return features.mean(dim=0)


def build_text_features(args, clip_model, dataset, classnames, device):
    cache_path = get_cache_path(args, dataset)
    if args.cache_text and os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            cached = pickle.load(handle)
        return cached.to(device)

    os.makedirs(args.cache_path, exist_ok=True)
    prompt_groups = [_class_prompts(name, args.use_templates) for name in classnames]
    if args.use_descriptors:
        descriptor_groups = _descriptor_prompts(dataset, classnames)
        prompt_groups = [base + desc for base, desc in zip(prompt_groups, descriptor_groups)]

    text_features = []
    for prompts in prompt_groups:
        text_features.append(_encode_prompt_group(clip_model, prompts, device))
    text_features = F.normalize(torch.stack(text_features, dim=0), dim=-1).detach()

    if args.cache_text:
        with open(cache_path, "wb") as handle:
            pickle.dump(text_features.cpu(), handle)
    return text_features.to(device)
