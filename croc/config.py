from dataclasses import dataclass


@dataclass(frozen=True)
class CROCConfig:
    method: str = "croc"
    n_views: int = 64
    alpha: int = 3
    theta: float = 0.995
    rho: float = 0.1
    num_steps: int = 1
    lr: float = 5e-3
    softsort_tau: float = 1.0
    num_alt_norm_steps: int = 1
    eps: float = 1e-6
    encode_chunk: int = 128
    disable_feature_update: bool = False
    disable_exclusion: bool = False
    debug: bool = False

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            method=str(cfg.get("method", "croc")).lower(),
            n_views=int(cfg.get("n_views", 64)),
            alpha=int(cfg.get("alpha", 3)),
            theta=float(cfg.get("theta", 0.995)),
            rho=float(cfg.get("rho", 0.1)),
            num_steps=int(cfg.get("num_steps", 1)),
            lr=float(cfg.get("lr", 5e-3)),
            softsort_tau=float(cfg.get("softsort_tau", 1.0)),
            num_alt_norm_steps=int(cfg.get("num_alt_norm_steps", 1)),
            eps=float(cfg.get("eps", 1e-6)),
            encode_chunk=int(cfg.get("encode_chunk", 128)),
            disable_feature_update=bool(cfg.get("disable_feature_update", False)),
            disable_exclusion=bool(cfg.get("disable_exclusion", False)),
            debug=bool(cfg.get("debug", False)),
        )

    def validate(self):
        if self.method not in {"croc", "clip"}:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.n_views < 1:
            raise ValueError("n_views must be at least 1")
        if self.alpha < 2:
            raise ValueError("alpha must be at least 2")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.softsort_tau <= 0.0:
            raise ValueError("softsort_tau must be positive")
        if self.num_alt_norm_steps < 1:
            raise ValueError("num_alt_norm_steps must be at least 1")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.encode_chunk < 1:
            raise ValueError("encode_chunk must be at least 1")
