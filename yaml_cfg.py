import yaml


DEFAULTS = {
    "method": "croc",
    "n_views": 64,
    "alpha": 3,
    "theta": 0.995,
    "rho": 0.1,
    "num_steps": 1,
    "lr": 5e-3,
    "softsort_tau": 1.0,
    "num_alt_norm_steps": 1,
    "eps": 1e-6,
    "encode_chunk": 128,
    "disable_feature_update": False,
    "disable_exclusion": False,
    "debug": False,
}


def load_tta_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = DEFAULTS.copy()
    cfg.update(raw)
    return cfg
