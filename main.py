import argparse
import csv
import json
import os
from datetime import datetime

from data.cls_to_names import (
    aircraft_classes,
    caltech101_classes,
    cars_classes,
    dtd_classes,
    eurosat_classes,
    flower102_classes,
    food101_classes,
    pets_classes,
    sun397_classes,
    ucf101_classes,
)
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from data.imagnet_prompts import imagenet_classes


BENCHMARK_DATASETS = [
    "I", "A", "V", "R", "K", "caltech101", "pets", "cars", "flower102",
    "food101", "aircraft", "sun397", "dtd", "eurosat", "ucf101",
]

CLASSNAME_MAP = {
    "aircraft": aircraft_classes,
    "caltech101": caltech101_classes,
    "cars": cars_classes,
    "dtd": dtd_classes,
    "eurosat": eurosat_classes,
    "flower102": flower102_classes,
    "food101": food101_classes,
    "pets": pets_classes,
    "sun397": sun397_classes,
    "ucf101": ucf101_classes,
}


def _parse_datasets(args):
    raw = args.datasets if args.datasets else args.dataset
    datasets = [item.strip() for item in raw.replace(",", "/").split("/") if item.strip()]
    unknown = [dataset for dataset in datasets if dataset not in BENCHMARK_DATASETS]
    if unknown:
        raise ValueError(f"Unsupported dataset id(s): {', '.join(unknown)}")
    return datasets


def _classnames_for_dataset(dataset):
    if dataset in {"I", "K"}:
        return imagenet_classes
    if dataset == "A":
        return [imagenet_classes[index] for index in imagenet_a_mask]
    if dataset == "R":
        return [name for index, name in enumerate(imagenet_classes) if imagenet_r_mask[index]]
    if dataset == "V":
        return [imagenet_classes[index] for index in imagenet_v_mask]
    return CLASSNAME_MAP[dataset]


def _config_path(args, dataset):
    return args.config or os.path.join(args.config_dir, f"{dataset}.yaml")


def _merge_cli_config(args, cfg):
    merged = dict(cfg)
    for key in (
        "method", "n_views", "alpha", "theta", "rho", "num_steps", "lr",
        "softsort_tau", "num_alt_norm_steps", "eps", "encode_chunk",
        "disable_feature_update", "disable_exclusion", "debug",
    ):
        merged[key] = getattr(args, key)
    return merged


def _build_transform(args):
    import torchvision.transforms as transforms

    from croc.transforms import build_eval_transform, build_multiview_transform

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    if args.n_views > 1:
        return build_multiview_transform(args.resolution, normalize, args.n_views, augmix=args.augmix)
    return build_eval_transform(args.resolution, normalize)


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _apply_prompt_type(args):
    args.use_templates = args.prompt_type == "templates"
    args.use_descriptors = args.prompt_type == "cupl"


def _write_result_csv(path, rows):
    fieldnames = [
        "method", "backbone", "dataset", "top1_acc", "num_samples", "n_views",
        "alpha", "theta", "rho", "num_steps", "lr",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def run(args):
    import torch
    import torch.backends.cudnn as cudnn

    from clip.custom_clip import build_frozen_clip
    from croc.eval import evaluate_croc
    from data.datautils import build_dataset
    from utils.tools import set_random_seed
    from yaml_cfg import load_tta_config

    if args.batch_size != 1:
        raise ValueError("CROC is episodic and requires --batch-size 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CROC benchmark evaluation requires a CUDA device")
    _apply_prompt_type(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    datasets = _parse_datasets(args)
    transform = _build_transform(args)
    run_tag = args.exp_name or f"{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    summary = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": args.method,
        "backbone": args.backbone,
        "prompt_type": args.prompt_type,
        "datasets": datasets,
        "results": {},
    }
    csv_rows = []
    model = None
    for dataset in datasets:
        classnames = _classnames_for_dataset(dataset)
        if model is None:
            model = build_frozen_clip(args, args.backbone, dataset, args.gpu, classnames).cuda(args.gpu)
        else:
            model.reset_classnames(args, classnames, dataset)
        model.eval()
        model.init_text_features()

        cfg = _merge_cli_config(args, load_tta_config(_config_path(args, dataset)))
        dataset_obj = build_dataset(dataset, transform, args.data_root, mode=args.split)
        loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=1,
            shuffle=args.shuffle,
            num_workers=args.workers,
            pin_memory=True,
        )
        result = evaluate_croc(loader, model, args, cfg, return_details=True)
        summary["results"][dataset] = result
        csv_rows.append(
            {
                "method": args.method,
                "backbone": args.backbone,
                "dataset": dataset,
                "top1_acc": result["top1"],
                "num_samples": result["num_samples"],
                "n_views": args.n_views,
                "alpha": args.alpha,
                "theta": args.theta,
                "rho": args.rho,
                "num_steps": args.num_steps,
                "lr": args.lr,
            }
        )
        print(
            f"{dataset}: {args.method.upper()}@1={result['top1']:.2f} "
            f"feature_updates={result['num_feature_updates']} exclusions={result['num_exclusions']}"
        )
        del dataset_obj, loader

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_result_csv(os.path.join(run_dir, "results.csv"), csv_rows)
    print(f"Saved results to {run_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="CROC episodic test-time adaptation evaluation")
    parser.add_argument("--method", default="croc", choices=["croc", "clip"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="I")
    parser.add_argument("--datasets", default=None, help="Slash or comma separated dataset ids")
    parser.add_argument("--config", default=None)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--shuffle", default=False, type=_str_to_bool)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-views", default=64, type=int)
    parser.add_argument("--augmix", action="store_true")
    parser.add_argument("--limit-samples", default=0, type=int)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--cache-path", default="cache")
    parser.add_argument("--cache-text", action="store_true")
    parser.add_argument("--prompt-type", default="cupl", choices=["cupl", "simple", "templates"])
    parser.add_argument("--use-templates", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use-descriptors", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--encode-chunk", default=128, type=int)
    parser.add_argument("--alpha", default=3, type=int)
    parser.add_argument("--theta", default=0.995, type=float)
    parser.add_argument("--rho", default=0.1, type=float)
    parser.add_argument("--num-steps", default=1, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--softsort-tau", default=1.0, type=float)
    parser.add_argument("--num-alt-norm-steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--disable-feature-update", action="store_true")
    parser.add_argument("--disable-exclusion", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
