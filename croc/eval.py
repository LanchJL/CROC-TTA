import time
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.tools import AverageMeter, accuracy
from .adapter import CROCAdapter
from .config import CROCConfig


@torch.no_grad()
def _encode_in_chunks(model, images: torch.Tensor, chunk: int):
    encoded = []
    for start in range(0, images.size(0), chunk):
        encoded.append(model.encode_image(images[start : start + chunk]))
    return F.normalize(torch.cat(encoded, dim=0).float(), dim=-1)


def _single_sample_views(images, device_id: int):
    if isinstance(images, list):
        if any(view.size(0) != 1 for view in images):
            raise ValueError("CROC requires batch_size=1")
        return torch.cat([view.cuda(device_id, non_blocking=True) for view in images], dim=0)
    if images.ndim == 5:
        if images.size(0) != 1:
            raise ValueError("CROC requires batch_size=1")
        return images[0].cuda(device_id, non_blocking=True)
    if images.size(0) != 1:
        raise ValueError("CROC requires batch_size=1")
    return images.cuda(device_id, non_blocking=True)


def evaluate_croc(val_loader, model, args, cfg: Dict, return_details: bool = False):
    croc_cfg = CROCConfig.from_dict(cfg)
    croc_cfg.validate()
    if model.text_features is None:
        raise RuntimeError("Text features are not initialized")
    logit_scale = float(model.logit_scale.detach().exp().float().item())
    text_features = model.text_features.detach().float()
    adapter = CROCAdapter(logit_scale, text_features, croc_cfg)

    top1 = AverageMeter("CROC@1", ":6.2f")
    top5 = AverageMeter("CROC@5", ":6.2f")
    total_samples = 0
    total_time = 0.0
    entropy_sum = 0.0
    feature_updates = 0
    exclusions = 0
    limit_samples = int(getattr(args, "limit_samples", 0))

    progress = tqdm(val_loader, total=len(val_loader), desc="Validating", dynamic_ncols=True, leave=False)
    for images, target in progress:
        start = time.perf_counter()
        views = _single_sample_views(images, args.gpu)
        if views.size(0) != croc_cfg.n_views:
            raise ValueError(f"Expected {croc_cfg.n_views} views, received {views.size(0)}")
        target = target.cuda(args.gpu, non_blocking=True)
        features = _encode_in_chunks(model, views, croc_cfg.encode_chunk)
        output = adapter.predict(features)
        posterior = output["posterior"].unsqueeze(0)
        acc1, acc5 = accuracy(posterior, target, topk=(1, min(5, posterior.size(-1))))
        top1.update(acc1.item())
        top5.update(acc5.item())
        total_samples += 1
        total_time += time.perf_counter() - start
        entropy_sum += float(output["top_alpha_entropy"].item())
        feature_updates += int(output["feature_updated"].item())
        exclusions += int(output["used_exclusion"].item())
        progress.set_description(f"Validating | CROC@1 {top1.avg:.2f}")
        if limit_samples > 0 and total_samples >= limit_samples:
            break

    result = {
        "top1": float(top1.avg),
        "top5": float(top5.avg),
        "num_classes": int(adapter.num_classes),
        "num_samples": int(total_samples),
        "mean_top_alpha_entropy": entropy_sum / max(total_samples, 1),
        "num_feature_updates": int(feature_updates),
        "num_exclusions": int(exclusions),
        "runtime_per_sample": total_time / max(total_samples, 1),
    }
    return result if return_details else [result["top1"], result["top5"]]
