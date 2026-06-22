# CROC: Confidently Ruling Out Uncertainty

PyTorch implementation of **CROC (Confidently Ruling Out Uncertainty)** for episodic test-time zero-shot generalization with CLIP.

CROC keeps the CLIP image and text encoders frozen. For each test image, it independently optimizes small feature shifts in embedding space, estimates uncertainty from sorted class probabilities, and rules out unlikely candidates using augmented views. No information is shared between test samples.

## Method

For each image, CROC performs the following steps:

1. Create one deterministic primary view and multiple augmented views.
2. Encode all views with the frozen CLIP image encoder.
3. Load frozen class text prototypes from CuPL descriptions or standard prompts.
4. Optimize per-sample visual and class-specific text shifts with a sort-based entropy loss.
5. Compute normalized top-α entropy on the primary prediction.
6. Return the primary prediction when it is confident.
7. Otherwise, construct sorting-induced exclusionary distributions from augmented views.
8. Apply alternating column and row normalization to suppress unlikely candidates.
9. Discard all feature shifts before processing the next image.

The implementation does not use a memory bank, cache queue, historical test features, target labels, target class priors, or persistent parameter updates.

## Repository Layout

```text
clip/       Frozen CLIP model and tokenizer
configs/    Dataset-specific CROC configurations
croc/       CROC adapter, operations, evaluation, transforms, and checks
data/       Dataset loaders, class names, and ImageNet mappings
model/      Prompt loading and text-prototype construction
prompts/    CuPL-style public class descriptions
scripts/    Natural-shift and cross-dataset launchers
utils/      Evaluation utilities
eval.py     Evaluation entry point
main.py     CLI and benchmark orchestration
```

## Installation

Python 3.9 or newer is recommended.

```bash
pip install torch torchvision ftfy regex tqdm pyyaml pillow numpy
```

The bundled CLIP loader downloads pretrained weights when they are not already available in the local CLIP cache.

## Datasets

Set up datasets outside the repository and pass their parent directory through `--data-root`.

```text
<DATA_ROOT>/
├── imagenet/val/
├── imagenet-a/images/
├── imagenetv2/images/
├── imagenet-r/images/
├── ImageNet-Sketch/images/
├── SUN397/
├── fgvc_aircraft/
├── eurosat/
├── stanford_cars/
├── food-101/
├── oxford_pets/
├── Flower102/
├── caltech-101/
├── dtd/
└── ucf101/
```

Datasets and model weights are not included in this repository.

## Quick Start

Evaluate one dataset with the default CROC configuration:

```bash
python eval.py \
  --method croc \
  --backbone ViT-B/16 \
  --dataset A \
  --data-root <DATA_ROOT> \
  --prompt-type cupl \
  --batch-size 1
```

CROC is episodic and therefore requires `--batch-size 1`.

Run a short smoke test with a local dataset:

```bash
python eval.py \
  --method croc \
  --dataset I \
  --data-root <DATA_ROOT> \
  --limit-samples 8
```

## Default Configuration

```text
backbone: ViT-B/16
prompt_type: cupl
n_views: 64
alpha: 3
theta: 0.995
rho: 0.1
num_steps: 1
lr: 0.005
softsort_tau: 1.0
num_alt_norm_steps: 1
```

Override these values with `--n-views`, `--alpha`, `--theta`, `--rho`, `--num-steps`, `--lr`, `--softsort-tau`, and `--num-alt-norm-steps`.

Use `--cache-text` to cache frozen text prototypes locally. Cache files remain excluded from version control.

## Benchmarks

Natural distribution shifts over ImageNet, ImageNet-A, ImageNet-V2, ImageNet-R, and ImageNet-Sketch:

```bash
bash scripts/run_croc_natural_shifts.sh --data-root <DATA_ROOT>
```

Cross-dataset generalization over SUN397, FGVC-Aircraft, EuroSAT, StanfordCars, Food101, OxfordPets, Flowers102, Caltech101, DTD, and UCF101:

```bash
bash scripts/run_croc_cross_dataset.sh --data-root <DATA_ROOT>
```

The natural-shift launcher reports the overall average and OOD average. The cross-dataset launcher reports the average over all ten datasets.

## Ablations

Frozen CLIP baseline:

```bash
python eval.py --method clip --dataset I --data-root <DATA_ROOT>
```

CROC without the sort-based feature update:

```bash
python eval.py --method croc --disable-feature-update --dataset I --data-root <DATA_ROOT>
```

CROC without exclusion inference:

```bash
python eval.py --method croc --disable-exclusion --dataset I --data-root <DATA_ROOT>
```

Enabling both disable flags is equivalent to the frozen CLIP prediction path.

## Outputs

Each run writes `results.json` and `results.csv` under the selected output directory. Per-dataset CSV files contain:

```text
method,backbone,dataset,top1_acc,num_samples,n_views,alpha,theta,rho,num_steps,lr
```

Generated outputs, caches, logs, model weights, and local datasets are ignored by Git.

## Validation

Run the CPU-only mathematical and episodic-state checks:

```bash
python -m croc.sanity
```

Use `--debug` during evaluation to enable additional posterior validity checks.

## Privacy and Reproducibility

- Keep datasets, checkpoints, caches, and experiment outputs outside version control.
- Use placeholders such as `<DATA_ROOT>` in commands and documentation.
- Do not add absolute local paths, credentials, API keys, access tokens, or machine-specific configuration.
- Review `git status --short` before publishing the repository.
