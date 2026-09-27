"""
knowledge_gated.py

Context-aware MemeBLIP2 model for MUTE fine-tuning.

This model wraps the updated CLIP-text MemeBLIP2 architecture and adds
cross-attention over external knowledge embeddings.

Used for:
    context_only
    context_distillation

Core idea:
    meme_tokens = [image_token, text_token]
    Q = meme_tokens
    K = knowledge tokens
    V = knowledge tokens

Then the cross-attended knowledge update is injected through a 
residual gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from memeblip2 import FeatureAdapter, LinearProjector, MemeBLIP2

try:
    from config import (
        BLIP2_MODEL_NAME,
        BETA,
        CLIP_TEXT_DIM,
        DROPOUT,
        FUSION_TYPE,
        KNOWLEDGE_DIM,
        NUM_CLASSES,
        NUM_HEADS,
        PROJ_LAYERS,
        SHARED_DIM,
        TRAIN_BASE,
        USE_RESIDUAL_GATE,
    )
except Exception:
    BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
    BETA = 0.0
    CLIP_TEXT_DIM = 512
    DROPOUT = 0.5
    FUSION_TYPE = "concat"
    KNOWLEDGE_DIM = 384
    NUM_CLASSES = 2
    NUM_HEADS = 8
    PROJ_LAYERS = 1
    SHARED_DIM = 512
    TRAIN_BASE = True
    USE_RESIDUAL_GATE = True


class CrossAttentionKnowledgeMemeBLIP2(nn.Module):
    """
    MemeBLIP2 + cross-attention over external knowledge embeddings.
    """

    def __init__(
        self,
        base_model: MemeBLIP2 | None = None,
        blip2_name: str = BLIP2_MODEL_NAME,
        shared_dim: int = SHARED_DIM,
        clip_text_dim: int = CLIP_TEXT_DIM,
        proj_layers: int = PROJ_LAYERS,
        dropout: float = DROPOUT,
        beta: float = BETA,
        num_classes: int = NUM_CLASSES,
        knowledge_dim: int = KNOWLEDGE_DIM,
        train_base: bool = TRAIN_BASE,
        num_heads: int = NUM_HEADS,
        use_residual_gate: bool = USE_RESIDUAL_GATE,
        fusion_type: str = FUSION_TYPE,
    ):
        super().__init__()

        self.base = base_model or MemeBLIP2(
            blip2_name=blip2_name,
            shared_dim=shared_dim,
            clip_text_dim=clip_text_dim,
            proj_layers=proj_layers,
            dropout=dropout,
            beta=beta,
            num_classes=num_classes,
            fusion_type=fusion_type,
        )

        self.shared_dim = self.base.shared_dim
        self.knowledge_dim = int(knowledge_dim)
        self.use_residual_gate = bool(use_residual_gate)

        if self.shared_dim % num_heads != 0:
            raise ValueError(
                f"shared_dim={self.shared_dim} must be divisible by num_heads={num_heads}."
            )

        if not train_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # Project MiniLM knowledge embeddings into the MemeBLIP2 shared space.
        self.knowledge_projection = LinearProjector(
            input_dim=self.knowledge_dim,
            output_dim=self.shared_dim,
            num_layers=proj_layers,
            dropout=dropout,
        )

        self.knowledge_adapter = FeatureAdapter(
            dim=self.shared_dim,
            dropout=dropout,
        )

        self.knowledge_cross_attention = nn.MultiheadAttention(
            embed_dim=self.shared_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if self.use_residual_gate:
            self.residual_gate = nn.Sequential(
                nn.Linear(self.shared_dim * 2, self.shared_dim),
                nn.LayerNorm(self.shared_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.shared_dim, self.shared_dim),
            )

        self.fusion_norm = nn.LayerNorm(self.shared_dim)

        # The context branch produces one shared_dim feature, so it needs a
        # shared_dim -> shared_dim pre-output before reusing base.classifier.
        self.context_pre_output = nn.Sequential(
            nn.Linear(self.shared_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._init_knowledge_weights()

    # ========================================================
    # 1. Initialization
    # ========================================================

    def _init_knowledge_weights(self):
        modules = [
            self.knowledge_projection,
            self.knowledge_adapter,
            self.knowledge_cross_attention,
            self.context_pre_output,
        ]

        if self.use_residual_gate:
            modules.append(self.residual_gate)

        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ========================================================
    # 2. Meme token extraction
    # ========================================================

    def _extract_memeblip2_tokens(
        self,
        pixel_values: torch.Tensor,
        clip_text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create meme tokens from updated MemeBLIP2 features.

        Returns:
            [B, 2, shared_dim]
        """
        v_bar, u_bar = self.base.encode_project_adapt(
            pixel_values=pixel_values,
            clip_text_embedding=clip_text_embedding,
        )

        if v_bar.dim() == 2 and u_bar.dim() == 2:
            return torch.stack([v_bar, u_bar], dim=1)

        if v_bar.dim() == 3 and u_bar.dim() == 3:
            return torch.cat([v_bar, u_bar], dim=1)

        raise ValueError(
            f"Unexpected feature shapes: image={v_bar.shape}, text={u_bar.shape}"
        )

    # ========================================================
    # 3. Knowledge cross-attention
    # ========================================================

    def _apply_cross_attention_knowledge(
        self,
        meme_tokens: torch.Tensor,
        knowledge_embeddings: torch.Tensor,
        knowledge_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            meme_tokens:
                [B, L, shared_dim]

            knowledge_embeddings:
                [B, K, knowledge_dim] or [B, knowledge_dim]

            knowledge_mask:
                [B, K], True for valid knowledge, False for padding
        """
        if knowledge_embeddings.dim() == 2:
            knowledge_embeddings = knowledge_embeddings.unsqueeze(1)

        _, _, dim = knowledge_embeddings.shape
        if dim != self.knowledge_dim:
            raise ValueError(
                f"Expected knowledge_dim={self.knowledge_dim}, got {dim}."
            )

        knowledge_embeddings = knowledge_embeddings.to(
            device=meme_tokens.device,
            dtype=meme_tokens.dtype,
        )

        if knowledge_mask is not None:
            if knowledge_mask.dim() == 1:
                knowledge_mask = knowledge_mask.unsqueeze(1)
            knowledge_mask = knowledge_mask.to(device=meme_tokens.device, dtype=torch.bool)

        knowledge_tokens = self.knowledge_projection(knowledge_embeddings)
        knowledge_tokens = self.knowledge_adapter(knowledge_tokens)
        knowledge_tokens = F.normalize(knowledge_tokens, p=2, dim=-1)

        key_padding_mask = None
        no_valid_knowledge = None

        if knowledge_mask is not None:
            # PyTorch MultiheadAttention uses True = ignore key.
            key_padding_mask = ~knowledge_mask
            no_valid_knowledge = ~knowledge_mask.any(dim=1)

            # Avoid NaNs if a sample has no valid knowledge.
            if no_valid_knowledge.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[no_valid_knowledge, 0] = False

        knowledge_update, attention_weights = self.knowledge_cross_attention(
            query=meme_tokens,
            key=knowledge_tokens,
            value=knowledge_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        if knowledge_mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~knowledge_mask.unsqueeze(1),
                0.0,
            )

            attention_weights = attention_weights / attention_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)

            if no_valid_knowledge is not None and no_valid_knowledge.any():
                knowledge_update = knowledge_update.masked_fill(
                    no_valid_knowledge[:, None, None],
                    0.0,
                )
                attention_weights = attention_weights.masked_fill(
                    no_valid_knowledge[:, None, None],
                    0.0,
                )

        if self.use_residual_gate:
            meme_pooled = meme_tokens.mean(dim=1)

            if knowledge_mask is not None:
                mask_float = knowledge_mask.float().unsqueeze(-1)
                knowledge_pooled = (
                    knowledge_tokens * mask_float
                ).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
            else:
                knowledge_pooled = knowledge_tokens.mean(dim=1)

            gate_input = torch.cat([meme_pooled, knowledge_pooled], dim=-1)
            residual_gate = torch.sigmoid(self.residual_gate(gate_input))

            if no_valid_knowledge is not None and no_valid_knowledge.any():
                residual_gate = residual_gate.masked_fill(
                    no_valid_knowledge.unsqueeze(-1),
                    0.0,
                )

            fused_tokens = self.fusion_norm(
                meme_tokens + residual_gate.unsqueeze(1) * knowledge_update
            )
        else:
            residual_gate = None
            fused_tokens = self.fusion_norm(meme_tokens + knowledge_update)

        fused_feature = fused_tokens.mean(dim=1)

        return fused_feature, fused_tokens, attention_weights, residual_gate

    # ========================================================
    # 4. Forward
    # ========================================================

    def forward(
        self,
        pixel_values: torch.Tensor,
        clip_text_embedding: torch.Tensor,
        knowledge_embeddings: torch.Tensor | None = None,
        knowledge_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        meme_tokens = self._extract_memeblip2_tokens(
            pixel_values=pixel_values,
            clip_text_embedding=clip_text_embedding,
        )

        fused_tokens = meme_tokens
        attention_weights = None
        residual_gate = None

        if knowledge_embeddings is not None:
            (
                meme_feature,
                fused_tokens,
                attention_weights,
                residual_gate,
            ) = self._apply_cross_attention_knowledge(
                meme_tokens=meme_tokens,
                knowledge_embeddings=knowledge_embeddings,
                knowledge_mask=knowledge_mask,
            )
        else:
            meme_feature = meme_tokens.mean(dim=1)

        z = self.context_pre_output(meme_feature)
        logits = self.base.classifier(z)

        if return_dict:
            return {
                "logits": logits,
                "meme_feature": meme_feature,
                "meme_tokens": meme_tokens,
                "fused_tokens": fused_tokens,
                "knowledge_attention": attention_weights,
                "residual_gate": residual_gate,
            }

        return logits
