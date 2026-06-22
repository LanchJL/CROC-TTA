import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


BENCHMARKS = {
    "natural": {
        "datasets": ["I", "A", "V", "R", "K"],
        "labels": ["ImageNet", "ImageNet-A", "ImageNet-V2", "ImageNet-R", "ImageNet-Sketch"],
        "csv": "croc_natural_shifts.csv",
        "json": "croc_natural_shifts.json",
    },
    "cross": {
        "datasets": ["sun397", "aircraft", "eurosat", "cars", "food101", "pets", "flower102", "caltech101", "dtd", "ucf101"],
        "labels": ["SUN397", "Aircraft", "EuroSAT", "Cars", "Food101", "Pets", "Flowers", "Caltech101", "DTD", "UCF101"],
        "csv": "croc_cross_dataset.csv",
        "json": "croc_cross_dataset.json",
    },
}


def _run(args):
    benchmark = BENCHMARKS[args.benchmark]
    experiment = f"{args.method}_{args.benchmark}_{args.backbone.replace('/', '-')}"
    command = [
        sys.executable,
        "eval.py",
        "--method", args.method,
        "--backbone", args.backbone,
        "--datasets", "/".join(benchmark["datasets"]),
        "--data-root", args.data_root,
        "--output-dir", args.output_dir,
        "--exp-name", experiment,
        "--prompt-type", args.prompt_type,
        "--n-views", str(args.n_views),
        "--alpha", str(args.alpha),
        "--theta", str(args.theta),
        "--rho", str(args.rho),
        "--num-steps", str(args.num_steps),
        "--lr", str(args.lr),
        "--softsort-tau", str(args.softsort_tau),
        "--num-alt-norm-steps", str(args.num_alt_norm_steps),
        "--workers", str(args.workers),
        "--batch-size", "1",
        "--shuffle", "false",
    ]
    if args.limit_samples:
        command.extend(["--limit-samples", str(args.limit_samples)])
    if args.disable_feature_update:
        command.append("--disable-feature-update")
    if args.disable_exclusion:
        command.append("--disable-exclusion")
    subprocess.run(command, check=True)
    result_path = Path(args.output_dir) / experiment / "results.json"
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize(args, run):
    benchmark = BENCHMARKS[args.benchmark]
    row = {"Method": run["method"].upper(), "Backbone": run["backbone"]}
    values = []
    for dataset, label in zip(benchmark["datasets"], benchmark["labels"]):
        value = float(run["results"][dataset]["top1"])
        row[label] = f"{value:.2f}"
        values.append(value)
    row["Average"] = f"{sum(values) / len(values):.2f}"
    if args.benchmark == "natural":
        row["OOD Average"] = f"{sum(values[1:]) / len(values[1:]):.2f}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / benchmark["csv"]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    with (output_dir / benchmark["json"]).open("w", encoding="utf-8") as handle:
        json.dump({"benchmark": args.benchmark, "run": run, "summary": row}, handle, ensure_ascii=False, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="Run CROC benchmark suites")
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--method", default="croc", choices=["croc", "clip"])
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--prompt-type", default="cupl", choices=["cupl", "simple", "templates"])
    parser.add_argument("--n-views", default=64, type=int)
    parser.add_argument("--alpha", default=3, type=int)
    parser.add_argument("--theta", default=0.995, type=float)
    parser.add_argument("--rho", default=0.1, type=float)
    parser.add_argument("--num-steps", default=1, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--softsort-tau", default=1.0, type=float)
    parser.add_argument("--num-alt-norm-steps", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--limit-samples", default=0, type=int)
    parser.add_argument("--disable-feature-update", action="store_true")
    parser.add_argument("--disable-exclusion", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    _summarize(parsed, _run(parsed))
