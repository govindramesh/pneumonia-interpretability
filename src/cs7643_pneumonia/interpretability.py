from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import plot_faithfulness_curve, save_overlay_grid


def _require_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for interpretability.") from exc
    return torch


class GradCAM:
    def __init__(self, model: object, target_module: object) -> None:
        torch = _require_torch()
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self._forward_handle = target_module.register_forward_hook(self._capture_activations)
        self._backward_handle = target_module.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, module: object, inputs: tuple, output: object) -> None:
        self.activations = output.detach()

    def _capture_gradients(self, module: object, grad_inputs: tuple, grad_outputs: tuple) -> None:
        self.gradients = grad_outputs[0].detach()

    def __call__(self, image_tensor: object) -> np.ndarray:
        torch = _require_torch()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits.reshape(-1)[0]
        score.backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def tensor_to_display_image(image_tensor: object) -> np.ndarray:
    torch = _require_torch()
    image = image_tensor.detach().cpu().clone().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    return np.transpose(image.numpy(), (1, 2, 0))


def upsample_map(saliency_map: np.ndarray, height: int, width: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for saliency resizing.") from exc
    resized = Image.fromarray((saliency_map * 255).astype(np.uint8)).resize((width, height))
    return np.asarray(resized, dtype=np.float32) / 255.0


def mask_salient_region(image_tensor: object, saliency_map: np.ndarray, fraction: float) -> object:
    torch = _require_torch()
    masked = image_tensor.clone()
    _, _, height, width = masked.shape
    saliency = upsample_map(saliency_map, height, width)
    flat = saliency.reshape(-1)
    num_mask = max(1, int(flat.size * fraction))
    top_indices = np.argpartition(flat, -num_mask)[-num_mask:]
    mask = np.ones_like(flat, dtype=np.float32)
    mask[top_indices] = 0.0
    mask = torch.tensor(mask.reshape(height, width), dtype=masked.dtype, device=masked.device).unsqueeze(0)
    masked = masked * mask.unsqueeze(0)
    return masked


def deletion_curve(model: object, image_tensor: object, saliency_map: np.ndarray, steps: int) -> tuple[list[float], list[float]]:
    torch = _require_torch()
    fractions = np.linspace(0.0, 1.0, steps).tolist()
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for fraction in fractions:
            masked = image_tensor if fraction == 0 else mask_salient_region(image_tensor, saliency_map, fraction)
            probability = torch.sigmoid(model(masked)).reshape(-1)[0].item()
            scores.append(float(probability))
    return fractions, scores


def confidence_drop(model: object, image_tensor: object, saliency_map: np.ndarray, fraction: float) -> tuple[float, float, float]:
    torch = _require_torch()
    model.eval()
    with torch.no_grad():
        original = torch.sigmoid(model(image_tensor)).reshape(-1)[0].item()
        masked = mask_salient_region(image_tensor, saliency_map, fraction)
        masked_score = torch.sigmoid(model(masked)).reshape(-1)[0].item()
    return float(original), float(masked_score), float(original - masked_score)


def explain_single_image(model: object, image_tensor: object, method_name: str, gradcam: GradCAM | None = None) -> np.ndarray:
    if method_name == "gradcam":
        if gradcam is None:
            raise ValueError("Grad-CAM object is required for gradcam explanations.")
        return gradcam(image_tensor)
    if method_name == "dino":
        patch_map = model.extract_patch_similarity_map(image_tensor)
        return patch_map.squeeze(0).detach().cpu().numpy()
    raise ValueError(f"Unsupported interpretation method: {method_name}")


def save_explanation_bundle(model: object, image_tensor: object, row: dict[str, str], output_dir: str | Path, method_name: str, gradcam: GradCAM | None = None, curve_steps: int = 10, mask_fraction: float = 0.2) -> dict[str, float | str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    display_image = tensor_to_display_image(image_tensor)
    saliency_map = explain_single_image(model, image_tensor, method_name=method_name, gradcam=gradcam)
    saliency_map = upsample_map(saliency_map, display_image.shape[0], display_image.shape[1])
    image_id = row["image_id"]

    save_overlay_grid(
        base_image=display_image,
        saliency_map=saliency_map,
        output_path=output_dir / f"{image_id}_{method_name}.png",
        title=f"{method_name.upper()} - {image_id}",
    )

    original, masked, drop = confidence_drop(model, image_tensor, saliency_map, fraction=mask_fraction)
    fractions, scores = deletion_curve(model, image_tensor, saliency_map, steps=curve_steps)
    plot_faithfulness_curve(
        fractions=fractions,
        scores=scores,
        output_path=output_dir / f"{image_id}_{method_name}_deletion.png",
        title=f"Deletion Curve - {image_id}",
    )

    return {
        "image_id": image_id,
        "patient_id": row["patient_id"],
        "label": row["label"],
        "method": method_name,
        "original_probability": original,
        "masked_probability": masked,
        "confidence_drop": drop,
        "overlay_path": str((output_dir / f"{image_id}_{method_name}.png").resolve()),
    }
