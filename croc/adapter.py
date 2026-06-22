from typing import Dict

import torch
import torch.nn.functional as F

from .config import CROCConfig
from .ops import (
    alternating_normalize,
    normalized_entropy,
    softsort_probs,
    softsort_topk_probs,
    sort_loss,
    top_alpha_entropy,
)


class CROCAdapter:
    def __init__(self, logit_scale: float, text_features: torch.Tensor, config: CROCConfig):
        config.validate()
        self.cfg = config
        self.text_features = F.normalize(text_features.detach().float(), dim=-1)
        self.logit_scale = float(logit_scale)
        self.num_classes, self.dim = self.text_features.shape
        if self.cfg.alpha > self.num_classes:
            raise ValueError("alpha cannot exceed the number of classes")

    def _posterior(self, image_features, delta_x, delta_t):
        shifted_images = F.normalize(image_features + delta_x.unsqueeze(0), dim=-1)
        shifted_text = F.normalize(self.text_features + delta_t, dim=-1)
        logits = self.logit_scale * shifted_images @ shifted_text.t()
        return torch.softmax(logits, dim=-1)

    def _feature_update(self, image_features):
        delta_x = torch.zeros(self.dim, device=image_features.device, dtype=torch.float32, requires_grad=True)
        delta_t = torch.zeros_like(self.text_features, requires_grad=True)
        should_update = (
            self.cfg.method == "croc"
            and not self.cfg.disable_feature_update
            and self.cfg.num_steps > 0
        )
        if should_update:
            optimizer = torch.optim.AdamW([delta_x, delta_t], lr=self.cfg.lr)
            for _ in range(self.cfg.num_steps):
                posterior = self._posterior(image_features, delta_x, delta_t)
                rank_probabilities = softsort_topk_probs(
                    posterior,
                    alpha=self.cfg.alpha,
                    tau=self.cfg.softsort_tau,
                )
                loss = sort_loss(rank_probabilities, rho=self.cfg.rho, eps=self.cfg.eps)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return delta_x.detach(), delta_t.detach(), should_update

    def _exclude(self, posterior: torch.Tensor, primary_entropy: torch.Tensor):
        primary = posterior[0]
        _, candidates = torch.topk(primary, k=self.cfg.alpha)
        candidate_posterior = posterior[:, candidates]
        candidate_ranks = softsort_probs(candidate_posterior, tau=self.cfg.softsort_tau)
        exclusion_rows = candidate_ranks[1:, 1:, :].reshape(-1, self.cfg.alpha)
        if exclusion_rows.size(0) == 0:
            return primary, False
        row_entropies = normalized_entropy(exclusion_rows, eps=self.cfg.eps)
        evidence = exclusion_rows[row_entropies < primary_entropy]
        if evidence.size(0) == 0:
            return primary, False
        matrix = torch.cat([evidence, primary[candidates].unsqueeze(0)], dim=0)
        adjusted = alternating_normalize(
            matrix,
            steps=self.cfg.num_alt_norm_steps,
            eps=self.cfg.eps,
        )[-1]
        output = torch.zeros_like(primary)
        output[candidates] = adjusted
        return output, True

    def predict(self, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_features = F.normalize(image_features.detach().float(), dim=-1)
        if image_features.ndim != 2 or image_features.size(1) != self.dim:
            raise ValueError("image_features must have shape [views, embedding_dim]")
        delta_x, delta_t, feature_updated = self._feature_update(image_features)
        with torch.no_grad():
            posterior = self._posterior(image_features, delta_x, delta_t)
            primary = posterior[0]
            primary_entropy = top_alpha_entropy(primary, self.cfg.alpha, eps=self.cfg.eps)
            should_exclude = (
                self.cfg.method == "croc"
                and not self.cfg.disable_exclusion
                and image_features.size(0) > 1
                and float(primary_entropy.item()) >= self.cfg.theta
            )
            output, used_exclusion = self._exclude(posterior, primary_entropy) if should_exclude else (primary, False)
        if self.cfg.debug:
            self._validate_output(output)
        return {
            "posterior": output,
            "primary_posterior": primary,
            "top_alpha_entropy": primary_entropy,
            "feature_updated": torch.tensor(feature_updated, device=image_features.device),
            "used_exclusion": torch.tensor(used_exclusion, device=image_features.device),
            "delta_x_norm": delta_x.norm(),
            "delta_t_norm": delta_t.norm(),
        }

    def _validate_output(self, posterior):
        if not torch.isfinite(posterior).all():
            raise AssertionError("posterior contains non-finite values")
        if not torch.all(posterior >= 0):
            raise AssertionError("posterior contains negative values")
        if not torch.allclose(posterior.sum(), torch.tensor(1.0, device=posterior.device), atol=1e-4):
            raise AssertionError("posterior is not normalized")
