import torch
import torch.nn.functional as F

from .adapter import CROCAdapter
from .config import CROCConfig
from .ops import alternating_normalize, softsort_probs, softsort_topk_probs, sort_loss, top_alpha_entropy


def run_sanity_checks():
    probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], requires_grad=True)
    top_ranks = softsort_topk_probs(probabilities, alpha=2, tau=0.1)
    assert top_ranks.shape == (2, 2, 3)
    assert torch.allclose(top_ranks.sum(dim=-1), torch.ones(2, 2), atol=1e-6)
    full_ranks = softsort_probs(probabilities, tau=0.1)
    assert int(full_ranks[0, 0].argmax()) == 0
    assert int(full_ranks[0, 2].argmax()) == 2

    loss = sort_loss(top_ranks, rho=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert probabilities.grad is not None and torch.isfinite(probabilities.grad).all()

    matrix = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    normalized = alternating_normalize(matrix, steps=2)
    assert torch.allclose(normalized.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert float(top_alpha_entropy(torch.tensor([0.98, 0.01, 0.01]), 3)) < 0.2

    text = F.normalize(torch.eye(4), dim=-1)
    views = F.normalize(
        torch.tensor(
            [
                [1.0, 0.9, 0.0, 0.0],
                [0.8, 1.0, 0.1, 0.0],
                [1.0, 0.7, 0.2, 0.0],
                [0.7, 1.0, 0.0, 0.2],
            ]
        ),
        dim=-1,
    )
    base_text = text.clone()
    clip = CROCAdapter(10.0, text, CROCConfig(method="clip", alpha=3, n_views=4, debug=True))
    clip_output = clip.predict(views)
    assert not bool(clip_output["feature_updated"])
    assert not bool(clip_output["used_exclusion"])

    no_update = CROCAdapter(
        10.0,
        text,
        CROCConfig(alpha=3, n_views=4, theta=0.0, disable_feature_update=True, debug=True),
    )
    no_update.predict(views)

    exclusion_adapter = CROCAdapter(
        10.0,
        text,
        CROCConfig(alpha=3, n_views=4, softsort_tau=0.01, debug=True),
    )
    synthetic_posterior = torch.tensor(
        [
            [0.34, 0.33, 0.32, 0.01],
            [0.90, 0.08, 0.01, 0.01],
            [0.10, 0.85, 0.04, 0.01],
            [0.10, 0.05, 0.84, 0.01],
        ]
    )
    excluded, used_exclusion = exclusion_adapter._exclude(synthetic_posterior, torch.tensor(1.0))
    assert used_exclusion
    assert float(excluded[3]) == 0.0
    assert torch.allclose(excluded.sum(), torch.tensor(1.0), atol=1e-6)
    fallback, used_exclusion = exclusion_adapter._exclude(synthetic_posterior, torch.tensor(0.0))
    assert not used_exclusion
    assert torch.equal(fallback, synthetic_posterior[0])

    no_exclusion = CROCAdapter(
        10.0,
        text,
        CROCConfig(alpha=3, n_views=4, disable_exclusion=True, debug=True),
    )
    updated = no_exclusion.predict(views)
    assert bool(updated["feature_updated"])
    assert float(updated["delta_x_norm"]) > 0.0 or float(updated["delta_t_norm"]) > 0.0
    assert torch.equal(text, base_text)

    first = no_exclusion.predict(views)
    second = no_exclusion.predict(views)
    assert torch.allclose(first["posterior"], second["posterior"], atol=1e-6)


if __name__ == "__main__":
    run_sanity_checks()
