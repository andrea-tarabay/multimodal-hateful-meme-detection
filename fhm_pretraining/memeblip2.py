"""MemeBLIP2: A Lightweight Multimodal System to Detect Harmful Memes.

Updated architecture:
  Image -> BLIP-2 Vision Encoder (frozen) -> Linear Projector -> Feature Adapter -> \
  Multimodal Fusion (concat by default) -> Pre-Output DenseNN -> MLP Classifier
  Text  -> Precomputed CLIP Text Embedding -> Linear Projector -> Feature Adapter -> /

Why this update:
  The previous version created a randomly initialized text embedding layer and
  passed it through a frozen BLIP-2 Q-Former. This made the text branch learn
  language semantics from scratch. This version instead consumes precomputed
  CLIP text embeddings from input_and_context.py / precompute_inputs.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Blip2Model

try:
    from config import CLIP_TEXT_DIM, FUSION_TYPE, DROPOUT
except Exception:
    CLIP_TEXT_DIM = 512
    FUSION_TYPE = "concat"
    DROPOUT = 0.3


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class LinearProjector(nn.Module):
    """Multi-layer linear projection with residual connections (Section 3.2).

    Layer 0:   x^(0) = Dropout(ReLU(LayerNorm(W_0 x + b_0)))
    Layer l>0: x^(l) = Dropout(ReLU(LayerNorm(W_l x^(l-1) + b_l))) + x^(l-1)
    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.fc = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for l in range(num_layers):
            in_d = input_dim if l == 0 else output_dim
            self.fc.append(nn.Linear(in_d, output_dim))
            self.norms.append(nn.LayerNorm(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.relu(self.norms[0](self.fc[0](x))))
        for l in range(1, self.num_layers):
            h = self.dropout(F.relu(self.norms[l](self.fc[l](h)))) + h
        return h


class FeatureAdapter(nn.Module):
    """Bottleneck adapter with scaled residual connection (Section 3.3).

    x̃ = LayerNorm(x)
    h = W_2 · GELU(W_1 · x̃)    [W_1: c→c̃, W_2: c̃→c, c̃ = max(16, ⌊c/r⌋)]
    ĥ = Dropout(h)
    x' = x + α · ĥ              [α learnable, init 0.1]
    y  = LayerNorm(x')
    """

    def __init__(self, dim: int, reduction_factor: float = 1.5, dropout: float = 0.1):
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


# ---------------------------------------------------------------------------
# Full MemeBLIP2 model
# ---------------------------------------------------------------------------

class MemeBLIP2(nn.Module):
    """Full MemeBLIP2 pipeline with precomputed CLIP text embeddings.

    Args:
        blip2_name:     HuggingFace model id for BLIP-2 vision encoder.
        shared_dim:     Shared embedding dimension after projection.
        clip_text_dim:  Dimension of the precomputed CLIP text embedding.
        proj_layers:    Number of layers in each linear projector.
        dropout:        Dropout probability used throughout trainable modules.
        beta:           Adapter mix ratio in [0, 1].
        num_classes:    Number of output classes.
        fusion_type:    "concat" or "multiply". Default is "concat".
    """

    def __init__(
        self,
        blip2_name: str = "Salesforce/blip2-opt-2.7b",
        shared_dim: int = 1024,
        clip_text_dim: int = CLIP_TEXT_DIM,
        proj_layers: int = 2,
        dropout: float = DROPOUT,
        beta: float = 0.5,
        num_classes: int = 2,
        fusion_type: str = FUSION_TYPE,
    ):
        super().__init__()
        self.beta = beta
        self.shared_dim = shared_dim
        self.clip_text_dim = clip_text_dim
        self.fusion_type = fusion_type

        if self.fusion_type not in {"concat", "multiply"}:
            raise ValueError("fusion_type must be either 'concat' or 'multiply'.")

        # ----- Frozen BLIP-2 vision encoder ----------------------------------
        blip2 = Blip2Model.from_pretrained(blip2_name, torch_dtype=torch.float16)
        self.vision_model = blip2.vision_model       # ViT-g/14, output 1408-dim
        del blip2
        torch.cuda.empty_cache()

        for p in self.vision_model.parameters():
            p.requires_grad = False

        # BLIP-2 ViT-g/14 output dim and CLIP text embedding dim
        img_dim = 1408
        txt_dim = clip_text_dim

        # ----- Trainable modules ---------------------------------------------
        # Section 3.2: linear projectors -> shared space
        self.image_projection = LinearProjector(img_dim, shared_dim, proj_layers, dropout)
        self.text_projection = LinearProjector(txt_dim, shared_dim, proj_layers, dropout)

        # Section 3.3: feature adapters
        self.image_adapter = FeatureAdapter(shared_dim, dropout=dropout)
        self.text_adapter = FeatureAdapter(shared_dim, dropout=dropout)

        # Section 3.5: pre-output DenseNN (Linear -> LayerNorm -> ReLU -> Dropout)
        pre_output_input_dim = shared_dim * 2 if self.fusion_type == "concat" else shared_dim
        self.pre_output = nn.Sequential(
            nn.Linear(pre_output_input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Section 3.6: MLP-based classifier (LayerNorm -> GELU -> Dropout -> Linear)
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in (
            self.image_projection,
            self.text_projection,
            self.image_adapter,
            self.text_adapter,
            self.pre_output,
            self.classifier,
        ):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embedding from frozen BLIP-2 ViT, shape (B, 1408)."""
        vision_dtype = next(self.vision_model.parameters()).dtype
        pv = pixel_values.to(dtype=vision_dtype)
        out = self.vision_model(pixel_values=pv)
        return out.last_hidden_state[:, 0, :].float()

    def _encode_text(self, clip_text_embedding: torch.Tensor) -> torch.Tensor:
        """Return precomputed CLIP text embedding, shape (B, clip_text_dim)."""
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
        """Encode image/text, project to shared space, adapt, and normalize.

        Returns:
            v_bar: normalized image feature, shape [B, shared_dim]
            u_bar: normalized text feature, shape [B, shared_dim]
        """
        v = self._encode_image(pixel_values)
        u = self._encode_text(clip_text_embedding)

        v_proj = self.image_projection(v)
        u_proj = self.text_projection(u)

        v_adapted = self.beta * self.image_adapter(v_proj) + (1 - self.beta) * v_proj
        u_adapted = self.beta * self.text_adapter(u_proj) + (1 - self.beta) * u_proj

        v_bar = F.normalize(v_adapted, p=2, dim=-1)
        u_bar = F.normalize(u_adapted, p=2, dim=-1)

        return v_bar, u_bar

    def fuse_features(self, v_bar: torch.Tensor, u_bar: torch.Tensor) -> torch.Tensor:
        """Fuse normalized image and text features."""
        if self.fusion_type == "concat":
            return torch.cat([v_bar, u_bar], dim=-1)
        if self.fusion_type == "multiply":
            return v_bar * u_bar
        raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        clip_text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # 1-3. Frozen image encoding + precomputed CLIP text embedding,
        #      then projection/adaptation/normalization.
        v_bar, u_bar = self.encode_project_adapt(
            pixel_values=pixel_values,
            clip_text_embedding=clip_text_embedding,
        )

        # 4. Multimodal fusion
        f = self.fuse_features(v_bar, u_bar)

        # 5. Pre-output DenseNN
        z = self.pre_output(f)

        # 6. MLP classifier
        return self.classifier(z)
