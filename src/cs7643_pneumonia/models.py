from __future__ import annotations

from pathlib import Path


def _require_torch() -> tuple[object, object]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install dependencies from requirements.txt.") from exc
    return torch, nn


class SmallCNN:
    def __new__(cls) -> object:
        torch, nn = _require_torch()

        class _SmallCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=False),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=0.2),
                    nn.Linear(64, 1),
                )

            def forward(self, x: object) -> object:
                features = self.features(x)
                return self.classifier(features).squeeze(-1)

        return _SmallCNN()


class DINOv2LinearProbe:
    def __new__(cls, backbone_name: str = "dinov2_vits14", pretrained: bool = True, freeze_backbone: bool = True, image_size: int = 224, local_weights_path: str | None = None) -> object:
        torch, nn = _require_torch()

        class _DINOv2LinearProbe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = self._load_backbone(backbone_name, pretrained, local_weights_path)
                if freeze_backbone:
                    for parameter in self.backbone.parameters():
                        parameter.requires_grad = False
                feature_dim = self._infer_feature_dim(image_size)
                self.classifier = nn.Linear(feature_dim, 1)

            def _load_backbone(self, name: str, use_pretrained: bool, weights_path: str | None) -> object:
                if weights_path:
                    loaded = torch.load(weights_path, map_location="cpu")
                    if isinstance(loaded, nn.Module):
                        return loaded
                try:
                    return torch.hub.load("facebookresearch/dinov2", name, pretrained=use_pretrained)
                except Exception as exc:
                    raise RuntimeError(
                        "Unable to load DINOv2. This path may require internet access or cached torch.hub weights."
                    ) from exc

            def _extract_pooled_features(self, x: object) -> object:
                if hasattr(self.backbone, "forward_features"):
                    features = self.backbone.forward_features(x)
                    if isinstance(features, dict):
                        if "x_norm_clstoken" in features:
                            return features["x_norm_clstoken"]
                        if "x_prenorm" in features:
                            return features["x_prenorm"][:, 0]
                    if hasattr(features, "__getitem__"):
                        return features[:, 0]
                features = self.backbone(x)
                if hasattr(features, "ndim") and features.ndim == 3:
                    return features[:, 0]
                return features

            def _infer_feature_dim(self, image_size_value: int) -> int:
                self.backbone.eval()
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, image_size_value, image_size_value)
                    pooled = self._extract_pooled_features(dummy)
                return int(pooled.shape[-1])

            def forward(self, x: object) -> object:
                pooled = self._extract_pooled_features(x)
                return self.classifier(pooled).squeeze(-1)

            def extract_patch_similarity_map(self, x: object) -> object:
                features = self.backbone.forward_features(x)
                if not isinstance(features, dict) or "x_norm_patchtokens" not in features or "x_norm_clstoken" not in features:
                    raise RuntimeError("This DINOv2 backbone does not expose patch tokens for interpretability.")
                patch_tokens = features["x_norm_patchtokens"]
                cls_token = features["x_norm_clstoken"].unsqueeze(1)
                similarity = torch.nn.functional.cosine_similarity(patch_tokens, cls_token, dim=-1)
                num_patches = similarity.shape[-1]
                side = int(num_patches ** 0.5)
                return similarity.view(-1, side, side)

            def extract_attention_rollout_map(self, x: object) -> object:
                if not hasattr(self.backbone, "blocks"):
                    raise RuntimeError("This DINOv2 backbone does not expose transformer blocks for attention rollout.")

                captured_attentions: list[object] = []
                hooks = []

                def _capture_attention(module: object, inputs: tuple, output: object) -> None:
                    tokens = inputs[0]
                    if not hasattr(module, "qkv") or not hasattr(module, "num_heads"):
                        raise RuntimeError("Attention module does not expose qkv/num_heads required for rollout.")
                    batch_size, num_tokens, channels = tokens.shape
                    head_dim = channels // module.num_heads
                    qkv = (
                        module.qkv(tokens)
                        .reshape(batch_size, num_tokens, 3, module.num_heads, head_dim)
                        .permute(2, 0, 3, 1, 4)
                    )
                    queries, keys = qkv[0], qkv[1]
                    attention = (queries @ keys.transpose(-2, -1)) * getattr(module, "scale", head_dim ** -0.5)
                    attention = attention.softmax(dim=-1)
                    captured_attentions.append(attention.detach())

                for block in self.backbone.blocks:
                    hooks.append(block.attn.register_forward_hook(_capture_attention))

                try:
                    _ = self.backbone.forward_features(x)
                finally:
                    for hook in hooks:
                        hook.remove()

                if not captured_attentions:
                    raise RuntimeError("No attention maps were captured for DINO attention rollout.")

                device = captured_attentions[0].device
                batch_size = captured_attentions[0].shape[0]
                num_tokens = captured_attentions[0].shape[-1]
                rollout = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

                for attention in captured_attentions:
                    fused = attention.mean(dim=1)
                    fused = fused + torch.eye(num_tokens, device=device).unsqueeze(0)
                    fused = fused / fused.sum(dim=-1, keepdim=True)
                    rollout = fused @ rollout

                cls_to_patches = rollout[:, 0, 1:]
                side = int(cls_to_patches.shape[-1] ** 0.5)
                return cls_to_patches.view(batch_size, side, side)

        return _DINOv2LinearProbe()


def build_model(model_config: object, image_size: int) -> object:
    torch, nn = _require_torch()
    name = model_config.name.lower()
    if name == "cnn":
        return SmallCNN()
    if name == "resnet50":
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:
            raise RuntimeError("torchvision is required to build ResNet-50.") from exc
        weights = ResNet50_Weights.DEFAULT if model_config.pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        return model
    if name == "dinov2_linear":
        return DINOv2LinearProbe(
            backbone_name="dinov2_vits14",
            pretrained=model_config.pretrained,
            freeze_backbone=model_config.freeze_backbone,
            image_size=image_size,
            local_weights_path=model_config.local_weights_path,
        )
    raise ValueError(f"Unsupported model: {model_config.name}")


def resolve_module(model: object, module_path: str) -> object:
    current = model
    for chunk in module_path.split("."):
        current = getattr(current, chunk)
    return current


def checkpoint_payload(model: object, config: dict, epoch: int, metrics: dict) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }


def save_checkpoint(model: object, config: dict, epoch: int, metrics: dict, path: str | Path) -> None:
    torch, _ = _require_torch()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, config, epoch, metrics), output_path)


def load_checkpoint(path: str | Path, model_config: object, image_size: int, device: object) -> tuple[object, dict]:
    torch, _ = _require_torch()
    payload = torch.load(path, map_location=device)
    model = build_model(model_config, image_size=image_size)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload
