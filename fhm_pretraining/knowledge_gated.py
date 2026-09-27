"""
knowledge_gated.py

Extends MemeBLIP2 with cross-attention over retrieved external knowledge.

The main class CrossAttentionKnowledgeMemeBLIP2 wraps a MemeBLIP2 base and adds:
    - a knowledge projection + adapter that maps knowledge embeddings into the
      shared feature space,
    - a multi-head cross-attention layer where meme tokens attend to knowledge
      tokens (Q = meme, K = V = knowledge),
    - a residual gate that controls how much knowledge is mixed in,
    - a separate pre-output DenseNN that operates on shared_dim (not 2*shared_dim)
      because the knowledge branch produces a pooled feature, not a concatenation.

Used by:
    train_all_modes.py  (modes: context_only, context_distillation)
    evaluate.py
    infer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from memeblip2 import MemeBLIP2, LinearProjector, FeatureAdapter

try:
    from config import CLIP_TEXT_DIM, FUSION_TYPE, DROPOUT
except Exception:
    CLIP_TEXT_DIM = 512
    FUSION_TYPE = "concat"
    DROPOUT = 0.3


class CrossAttentionKnowledgeMemeBLIP2(nn.Module):
    """MemeBLIP2 extended with cross-attention over retrieved knowledge.

    Args:
        base_model:         Pre-built MemeBLIP2 instance. If None, one is created
                            from the remaining kwargs.
        blip2_name:         HuggingFace model id for BLIP-2 vision encoder.
        shared_dim:         Shared embedding dimension (must be divisible by num_heads).
        clip_text_dim:      Dimension of the precomputed CLIP text embedding.
        proj_layers:        Number of layers in each linear projector.
        dropout:            Dropout probability used throughout trainable modules.
        beta:               Adapter mix ratio in [0, 1].
        num_classes:        Number of output classes.
        knowledge_dim:      Embedding dimension of each retrieved knowledge sentence
                            (384 for all-MiniLM-L6-v2).
        train_base:         If False, freeze all MemeBLIP2 base parameters and only
                            train the knowledge cross-attention branch.
        num_heads:          Number of attention heads for cross-attention.
        use_residual_gate:  If True, learn a per-sample gate scalar that controls
                            how much cross-attended knowledge is mixed into the meme
                            representation. If False, knowledge is always fully added.
        fusion_type:        "concat" or "multiply" (forwarded to MemeBLIP2 base).
    """

    def __init__(
        self,
        base_model: MemeBLIP2 = None,
        blip2_name: str = "Salesforce/blip2-opt-2.7b",
        shared_dim: int = 1024,
        clip_text_dim: int = CLIP_TEXT_DIM,
        proj_layers: int = 2,
        dropout: float = DROPOUT,
        beta: float = 0.5,
        num_classes: int = 2,
        knowledge_dim: int = 384,
        train_base: bool = True,
        num_heads: int = 8,
        use_residual_gate: bool = True,
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
        self.knowledge_dim = knowledge_dim
        self.use_residual_gate = use_residual_gate

        if self.shared_dim % num_heads != 0:
            raise ValueError(
                f"shared_dim={self.shared_dim} must be divisible by num_heads={num_heads}."
            )

        if not train_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # Project knowledge embeddings into the same space as MemeBLIP2 features.
        self.knowledge_projection = LinearProjector(
            input_dim=knowledge_dim,
            output_dim=self.shared_dim,
            num_layers=proj_layers,
            dropout=dropout,
        )

        self.knowledge_adapter = FeatureAdapter(
            dim=self.shared_dim,
            dropout=dropout,
        )

        # Full cross-attention:
        # Q = meme tokens
        # K = knowledge tokens
        # V = knowledge tokens
        self.knowledge_cross_attention = nn.MultiheadAttention(
            embed_dim=self.shared_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Controls how much cross-attended knowledge is added.
        if use_residual_gate:
            self.residual_gate = nn.Sequential(
                nn.Linear(self.shared_dim * 2, self.shared_dim),
                nn.LayerNorm(self.shared_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.shared_dim, self.shared_dim),
            )

        self.fusion_norm = nn.LayerNorm(self.shared_dim)

        # The base MemeBLIP2 pre_output may expect 2 * shared_dim when using
        # concat fusion. The knowledge branch pools cross-attended meme tokens
        # to shared_dim, so it needs its own shared_dim -> shared_dim pre-output.
        self.context_pre_output = nn.Sequential(
            nn.Linear(self.shared_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._init_knowledge_weights()

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

    def _extract_memeblip2_tokens(
        self,
        pixel_values: torch.Tensor,
        clip_text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Creates meme tokens h_mm.

        The updated base model uses:
            image feature: projected/adapted BLIP-2 vision feature [B, D]
            text feature : projected/adapted precomputed CLIP text feature [B, D]

        Then we create:
            meme_tokens = [image_token, text_token]
            shape: [B, 2, D]
        """

        v_bar, u_bar = self.base.encode_project_adapt(
            pixel_values=pixel_values,
            clip_text_embedding=clip_text_embedding,
        )

        if v_bar.dim() == 2 and u_bar.dim() == 2:
            meme_tokens = torch.stack([v_bar, u_bar], dim=1)  # [B, 2, D]
        elif v_bar.dim() == 3 and u_bar.dim() == 3:
            meme_tokens = torch.cat([v_bar, u_bar], dim=1)  # [B, L_img + L_text, D]
        else:
            raise ValueError(
                f"Unexpected feature shapes: image={v_bar.shape}, text={u_bar.shape}"
            )

        return meme_tokens

    def _apply_cross_attention_knowledge(
        self,
        meme_tokens: torch.Tensor,
        knowledge_embeddings: torch.Tensor,
        knowledge_mask: torch.Tensor = None,
    ):
        """
        meme_tokens:
            [B, L, shared_dim]

        knowledge_embeddings:
            [B, K, knowledge_dim] or [B, knowledge_dim]

        knowledge_mask:
            [B, K]
            True  = valid knowledge
            False = padding
        """

        if knowledge_embeddings.dim() == 2:
            knowledge_embeddings = knowledge_embeddings.unsqueeze(1)

        B, K, D = knowledge_embeddings.shape

        if D != self.knowledge_dim:
            raise ValueError(
                f"Expected knowledge_dim={self.knowledge_dim}, but got {D}."
            )

        knowledge_embeddings = knowledge_embeddings.to(
            device=meme_tokens.device,
            dtype=meme_tokens.dtype,
        )

        if knowledge_mask is not None:
            if knowledge_mask.dim() == 1:
                knowledge_mask = knowledge_mask.unsqueeze(1)

            knowledge_mask = knowledge_mask.to(
                device=meme_tokens.device,
                dtype=torch.bool,
            )

        # Encode/project knowledge into M_K.
        knowledge_tokens = self.knowledge_projection(knowledge_embeddings)
        knowledge_tokens = self.knowledge_adapter(knowledge_tokens)
        knowledge_tokens = F.normalize(knowledge_tokens, p=2, dim=-1)

        key_padding_mask = None
        no_valid_knowledge = None

        if knowledge_mask is not None:
            # PyTorch MultiheadAttention uses True = ignore this key.
            key_padding_mask = ~knowledge_mask
            no_valid_knowledge = ~knowledge_mask.any(dim=1)

            # Avoid NaNs if a sample has no valid knowledge.
            if no_valid_knowledge.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[no_valid_knowledge, 0] = False

        # ---------------------------------------------------------
        # CROSS-ATTENTION:
        # Q = meme_tokens
        # K = knowledge_tokens
        # V = knowledge_tokens
        # ---------------------------------------------------------
        knowledge_update, attention_weights = self.knowledge_cross_attention(
            query=meme_tokens,
            key=knowledge_tokens,
            value=knowledge_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # knowledge_update: [B, L, shared_dim]
        # attention_weights: [B, L, K]

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

        # ---------------------------------------------------------
        # Gate decides how much of knowledge_update is added.
        # ---------------------------------------------------------
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

        # Pool final sequence before classification.
        fused_feature = fused_tokens.mean(dim=1)

        return fused_feature, fused_tokens, attention_weights, residual_gate

    def forward(
        self,
        pixel_values: torch.Tensor,
        clip_text_embedding: torch.Tensor,
        knowledge_embeddings: torch.Tensor = None,
        knowledge_mask: torch.Tensor = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict:
        """
        Args:
            pixel_values:          [B, 3, H, W]
            clip_text_embedding:   [B, clip_text_dim]
            knowledge_embeddings:  [B, K, knowledge_dim] or None (skips knowledge branch)
            knowledge_mask:        [B, K] bool — True = valid, False = padding
            return_dict:           If True, return a dict with logits + intermediate
                                   tensors for inspection/debugging.

        Returns:
            logits [B, num_classes]  or  dict with "logits" and auxiliary keys.
        """
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
