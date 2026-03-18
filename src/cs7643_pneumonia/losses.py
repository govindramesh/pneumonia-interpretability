from __future__ import annotations


def build_loss(name: str, pos_weight: float | None = None) -> object:
    import torch

    loss_name = name.lower()
    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    if loss_name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("weighted_bce requires a positive-class weight.")
        tensor_weight = torch.tensor([pos_weight], dtype=torch.float32)
        return torch.nn.BCEWithLogitsLoss(pos_weight=tensor_weight)
    if loss_name == "focal":
        return FocalLoss()
    raise ValueError(f"Unsupported loss: {name}")


class FocalLoss:
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits: object, targets: object) -> object:
        import torch
        import torch.nn.functional as F

        targets = targets.view_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_factor = torch.where(targets == 1, torch.full_like(targets, self.alpha), torch.full_like(targets, 1 - self.alpha))
        loss = alpha_factor * ((1 - pt) ** self.gamma) * bce
        return loss.mean()
