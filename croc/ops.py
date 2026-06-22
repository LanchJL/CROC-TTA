import math

import torch


def softsort_topk_probs(probabilities: torch.Tensor, alpha: int, tau: float = 1.0) -> torch.Tensor:
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have shape [..., classes]")
    if alpha > probabilities.size(-1):
        raise ValueError("alpha cannot exceed the number of classes")
    sorted_values = torch.sort(probabilities, dim=-1, descending=True).values[..., :alpha]
    distances = torch.abs(sorted_values.unsqueeze(-1) - probabilities.unsqueeze(-2))
    return torch.softmax(-distances / tau, dim=-1)


def softsort_probs(probabilities: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return softsort_topk_probs(probabilities, probabilities.size(-1), tau=tau)


def entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities.clamp_min(eps))).sum(dim=-1)


def normalized_entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    classes = probabilities.size(-1)
    if classes <= 1:
        return torch.zeros(probabilities.shape[:-1], device=probabilities.device, dtype=probabilities.dtype)
    normalized = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    return entropy(normalized, eps=eps) / math.log(classes)


def top_alpha_entropy(probabilities: torch.Tensor, alpha: int, eps: float = 1e-6) -> torch.Tensor:
    top_values = torch.topk(probabilities, k=alpha, dim=-1).values
    return normalized_entropy(top_values, eps=eps)


def sort_loss(rank_probabilities: torch.Tensor, rho: float, eps: float = 1e-6) -> torch.Tensor:
    if rank_probabilities.ndim != 3:
        raise ValueError("rank_probabilities must have shape [views, ranks, classes]")
    num_views, num_ranks, _ = rank_probabilities.shape
    num_select = max(1, int(round(rho * num_views)))
    rank_entropies = entropy(rank_probabilities, eps=eps)
    losses = []
    for rank in range(num_ranks):
        selected = torch.topk(-rank_entropies[:, rank], k=num_select).indices
        mean_rank_distribution = rank_probabilities[selected, rank].mean(dim=0)
        losses.append(entropy(mean_rank_distribution, eps=eps))
    return torch.stack(losses).mean()


def alternating_normalize(matrix: torch.Tensor, steps: int, eps: float = 1e-6) -> torch.Tensor:
    normalized = matrix
    for _ in range(steps):
        normalized = normalized / normalized.mean(dim=0, keepdim=True).clamp_min(eps)
        normalized = normalized / normalized.sum(dim=1, keepdim=True).clamp_min(eps)
    return normalized
