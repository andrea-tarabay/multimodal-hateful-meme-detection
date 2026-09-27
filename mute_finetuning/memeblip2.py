"""
memeblip2.py

MemeBLIP2: lightweight multimodal meme classifier for MUTE fine-tuning.

Updated architecture, aligned with the first-stage FHM code:

    Image -> frozen BLIP-2 vision encoder -> LinearProjector -> FeatureAdapter -> \
    Text  -> precomputed CLIP text embedding -> LinearProjector -> FeatureAdapter -> \
    Fusion -> Pre-output MLP -> Classifier

Important update:
    The text branch consumes precomputed CLIP text embeddings from
    input_and_context.py / precompute_inputs.py. 
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Blip2Model

try:
    from config import (
        BLIP2_MODEL_NAME,
        CLIP_TEXT_DIM,
        DROPOUT,
        FUSION_TYPE,
        NUM_CLASSES,
        PROJ_LAYERS,
        SHARED_DIM,
        BETA,
    )
except Exception:
    BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
    CLIP_TEXT_DIM = 512
    DROPOUT = 0.5
    FUSION_TYPE = "concat"
    NUM_CLASSES = 2
    PROJ_LAYERS = 1
    SHARED_DIM = 512
    BETA = 0.0


# ============================================================
# 1. Submodules
# ============================================================

class LinearProjector(nn.Module):
    """
    Multi-layer linear projection with residual connections.

    Layer 0:
        Dropout(ReLU(LayerNorm(Wx + b)))

    Layer l > 0:
        Dropout(ReLU(LayerNorm(Wh + b))) + h
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = int(num_layers)
        self.fc = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for layer_idx in range(self.num_layers):
            in_dim = input_dim if layer_idx == 0 else output_dim
            self.fc.append(nn.Linear(in_dim, output_dim))
            self.norms.append(nn.LayerNorm(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.relu(self.norms[0](self.fc[0](x))))

        for layer_idx in range(1, self.num_layers):
            h = self.dropout(F.relu(self.norms[layer_idx](self.fc[layer_idx](h)))) + h

        return h


class FeatureAdapter(nn.Module):
    """
    Bottleneck adapter with scaled residual connection.
    """

    def __init__(
        self,
        dim: int,
        reduction_factor: float = 1.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        bottleneck = max(16, int(dim / reduction_factor))
        self.norm1 = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(F.gelu(self.down(self.norm1(x))))
        return self.norm2(x + self.scale * self.dropout(h))


# ============================================================
# 2. MemeBLIP2
# ============================================================

class MemeBLIP2(nn.Module):
    """
    Base MemeBLIP2 model with frozen BLIP-2 image encoder and precomputed CLIP
    text embeddings.

    Args:
        blip2_name:
            Hugging Face BLIP-2 checkpoint used for the vision encoder.

        shared_dim:
            Dimension of the common image/text feature space.

        clip_text_dim:
            Dimension of the cached CLIP text embedding.

        proj_layers:
            Number of projection layers.

        dropout:
            Dropout probability in trainable modules.

        beta:
            Adapter mix ratio in [0, 1]. beta=0 means use projected features
            only; beta=1 means use adapted features only.

        fusion_type:
            "concat" or "multiply".
    """

    def __init__(
        self,
        blip2_name: str = BLIP2_MODEL_NAME,
        shared_dim: int = SHARED_DIM,
        clip_text_dim: int = CLIP_TEXT_DIM,
        proj_layers: int = PROJ_LAYERS,
        dropout: float = DROPOUT,
        beta: float = BETA,
        num_classes: int = NUM_CLASSES,
        fusion_type: str = FUSION_TYPE,
    ):
        super().__init__()

        self.beta = float(beta)
        self.shared_dim = int(shared_dim)
        self.clip_text_dim = int(clip_text_dim)
        self.fusion_type = fusion_type

        if self.fusion_type not in {"concat", "multiply"}:
            raise ValueError("fusion_type must be either 'concat' or 'multiply'.")

        # ----------------------------------------------------
        # Frozen BLIP-2 vision encoder
        # ----------------------------------------------------
        blip2 = Blip2Model.from_pretrained(blip2_name, torch_dtype=torch.float16)
        self.vision_model = blip2.vision_model
        del blip2
        torch.cuda.empty_cache()

        for param in self.vision_model.parameters():
            param.requires_grad = False

        img_dim = 1408
        txt_dim = self.clip_text_dim

        # ----------------------------------------------------
        # Trainable image/text projectors and adapters
        # ----------------------------------------------------
        self.image_projection = LinearProjector(
            input_dim=img_dim,
            output_dim=self.shared_dim,
            num_layers=proj_layers,
            dropout=dropout,
        )
        self.text_projection = LinearProjector(
            input_dim=txt_dim,
            output_dim=self.shared_dim,
            num_layers=proj_layers,
            dropout=dropout,
        )

        self.image_adapter = FeatureAdapter(self.shared_dim, dropout=dropout)
        self.text_adapter = FeatureAdapter(self.shared_dim, dropout=dropout)

        # ----------------------------------------------------
        # Fusion and classifier
        # ----------------------------------------------------
        pre_output_input_dim = self.shared_dim * 2 if self.fusion_type == "concat" else self.shared_dim

        self.pre_output = nn.Sequential(
            nn.Linear(pre_output_input_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        modules = [
            self.image_projection,
            self.text_projection,
            self.image_adapter,
            self.text_adapter,
            self.pre_output,
            self.classifier,
        ]

        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ========================================================
    # 3. Encoder helpers
    # ========================================================

    @torch.no_grad()
    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Return CLS-token embedding from frozen BLIP-2 ViT.

        Output shape:
            [B, 1408]
        """
        vision_dtype = next(self.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :].float()

    def _encode_text(self, clip_text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Return cached CLIP text embedding.

        Expected shape:
            [B, CLIP_TEXT_DIM]
        """
        if clip_text_embedding.dim() != 2:
            raise ValueError(
                f"clip_text_embedding should have shape [B, {self.clip_text_dim}], "
                f"got {clip_text_embedding.shape}."
            )

        if clip_text_embedding.shape[-1] != self.clip_text_dim:
            raise ValueError(
                f"Expected clip_text_dim={self.clip_text_dim}, "
                f"got {clip_text_embedding.shape[-1]}."
            )

        return clip_text_embedding.float()

    def encode_project_adapt(
        self,
        pixel_values: torch.Tensor,
        clip_text_embedding: torch.Tensor,
    ):
        """
        Encode image/text, project to shared space, adapt, and normalize.

        Returns:
            v_bar: [B, shared_dim]
            u_bar: [B, shared_dim]
        """
        v = self._encode_image(pixel_values)
        u = self._encode_text(clip_text_embedding)

        v_proj = self.image_projection(v)
        u_proj = self.text_projection(u)

        v_adapted = self.beta * self.image_adapter(v_proj) + (1.0 - self.beta) * v_proj
        u_adapted = self.beta * self.text_adapter(u_proj) + (1.0 - self.beta) * u_proj

        v_bar = F.normalize(v_adapted, p=2, dim=-1)
        u_bar = F.normalize(u_adapted, p=2, dim=-1)

        return v_bar, u_bar

    def fuse_features(self, v_bar: torch.Tensor, u_bar: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([v_bar, u_bar], dim=-1)
        if self.fusion_type == "multiply":
            return v_bar * u_bar
        raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    # ========================================================
    # 4. Forward
    # ========================================================

    def forward(
        self,
        pixel_values: torch.Tensor,
        clip_text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        v_bar, u_bar = self.encode_project_adapt(
            pixel_values=pixel_values,
            clip_text_embedding=clip_text_embedding,
        )

        fused = self.fuse_features(v_bar, u_bar)
        z = self.pre_output(fused)
        logits = self.classifier(z)

        return logits
