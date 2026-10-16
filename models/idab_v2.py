"""
IDAB: Inconsistency Detection with Adaptive Bifurcation
Overall pipeline: Backbone -> GSPM -> GSEM -> GSFM -> GSAM -> GS-Former
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EfficientNetBackbone
from .perception import GSPM
from .gsem import GSEM
from .gsfm import GSFM
from .gsam import GSAM
from .gs_former import GSFormer


class IDAB(nn.Module):
    """
    Three core components of IDAB:
    1. Generation State Perception Module (GSPM)
    2. Generation State Enhancement & Fusion Module (GSEM + GSFM)
    3. Generation State Aggregation & Modeling Module (GSAM + GS-Former)
    """

    def __init__(
        self,
        num_classes=5,
        embed_size=256,
        num_heads=8,
        num_layers=2,
        num_tokens=5,
        token_dim=512,
        bilinear_d=32,
        dropout=0.5,
        transformer_dropout=0.3,
        texture_lambda=1.0,
        use_dual_head=False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.use_dual_head = use_dual_head
        Ct, Cg = 32, 160
        self.z_dim = bilinear_d * bilinear_d  # Z: [B, T, d*d]

        self.backbone = EfficientNetBackbone(texture_block_idx=3, global_block_idx=16)
        self.gspm = GSPM(in_channels=Cg, hidden=64)
        self.gsem = GSEM(lambda_init=texture_lambda)
        self.gsfm = GSFM(C_tex=Ct, C_glob=Cg, d=bilinear_d)
        self.gsam = GSAM(in_dim=self.z_dim, num_tokens=num_tokens)
        # Project tokens to embed_size (sequence length K) for token-level reasoning in GS-Former.
        self.projector = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Dropout(dropout),
            nn.Linear(self.z_dim, embed_size),
            nn.Dropout(dropout),
        )
        self.gs_former = GSFormer(
            embed_size=embed_size,
            num_heads=num_heads,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=transformer_dropout,
        )
        # Dual-head: add a BiLSTM after GSFM for state-difference supervision.
        if use_dual_head:
            # Project per-frame Z to embed_size for the BiLSTM.
            self.feat_proj = nn.Sequential(
                nn.LayerNorm(self.z_dim),
                nn.Linear(self.z_dim, embed_size),
            )
            self.temporal_lstm = nn.LSTM(
                input_size=embed_size,
                hidden_size=embed_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.h_norm = nn.LayerNorm(embed_size)
            # Frame-level head: input H = Z_proj + lstm_out [B, T, embed_size]
            self.frame_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_size, num_classes),
            )
            # Inconsistency head: pair_feat [B, T-1, 2*embed_size] -> 1
            self.inconsistency_head = nn.Sequential(
                nn.Linear(embed_size * 2, embed_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_size, 1),
            )

    def forward(self, x, return_frame_predictions=False, apply_sigmoid=True):
        """
        Args:
            x: (B, T, 3, H, W)
        Returns:
            Video-level logits; if use_dual_head=True, also returns
            frame_logits [B,T,C], inc_logits [B,T-1,1], delta_norm [B,T-1].
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        Fg, Ft = self.backbone(x_flat)
        Fg = Fg.view(B, T, *Fg.shape[1:])
        Ft = Ft.view(B, T, *Ft.shape[1:])

        # Align Fg to the spatial resolution of Ft.
        _, _, Cg, Hg, Wg = Fg.shape
        _, _, Ct, Ht, Wt = Ft.shape
        Fg_align = F.interpolate(
            Fg.view(B * T, Cg, Hg, Wg),
            size=(Ht, Wt),
            mode='bilinear',
            align_corners=False,
        ).view(B, T, Cg, Ht, Wt)

        # 1) GSPM: generate spatio-temporal state attention map A_st
        A_st = self.gspm(Fg_align)

        # 2) GSEM: texture enhancement
        F_tex_enh = self.gsem(Ft, A_st)

        # 3) GSFM: bilinear fusion
        Z = self.gsfm(F_tex_enh, Fg_align, A_st)

        # 4) GSAM: multi-attention aggregation over time (temporal tokenization).
        Z_map = Z.permute(0, 2, 1).unsqueeze(-1)
        A_temporal = A_st.mean(dim=(-2, -1)).view(B, T, 1, 1).permute(0, 2, 1, 3)
        tokens = self.gsam(Z_map, attention_map=A_temporal)

        # 5) Token projection -> GS-Former -> video-level logits
        proj = self.projector(tokens)
        logits = self.gs_former(proj)
        if apply_sigmoid:
            out = torch.sigmoid(logits)
        else:
            out = logits

        if self.use_dual_head:
            # Z [B, T, z_dim] -> projection -> BiLSTM -> residual -> LayerNorm -> H
            # (stabilizes scale to avoid exploding delta_norm)
            Z_proj = self.feat_proj(Z)
            lstm_out, _ = self.temporal_lstm(Z_proj)
            H = self.h_norm(Z_proj + lstm_out)
            H = torch.nan_to_num(H, nan=0.0, posinf=10.0, neginf=-10.0)
            # Head1: frame-level source classification
            frame_logits = self.frame_head(H)
            # State difference & Head2: inconsistency (pair_feat = [h_t, h_{t+1}])
            pair_feat = torch.cat([H[:, :-1], H[:, 1:]], dim=-1)
            inc_logits = self.inconsistency_head(pair_feat)
            delta_norm = torch.norm(H[:, 1:] - H[:, :-1], dim=-1)
            delta_norm = torch.nan_to_num(delta_norm, nan=0.0, posinf=5.0, neginf=0.0)
            if apply_sigmoid:
                frame_logits = torch.sigmoid(frame_logits)
                inc_logits = torch.sigmoid(inc_logits)
            return out, frame_logits, inc_logits, delta_norm

        if return_frame_predictions:
            frame_logits = []
            for t in range(T):
                Z_t_map = Z[:, t].view(B, -1, 1, 1)
                A_t = A_st[:, t].mean(dim=(-2, -1)).view(B, 1, 1, 1)
                tok_t = self.gsam(Z_t_map, attention_map=A_t)
                p_t = self.projector(tok_t)
                log_t = self.gs_former(p_t)
                frame_logits.append(log_t)
            frame_logits = torch.stack(frame_logits, dim=1)
            if apply_sigmoid:
                frame_logits = torch.sigmoid(frame_logits)
            return out, frame_logits
        return out

    def get_video_features(self, x):
        """
        Extract video-level feature vectors by mean-pooling the projector outputs over tokens.
        Useful for visualization such as t-SNE.
        Returns:
            (B, embed_size)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        Fg, Ft = self.backbone(x_flat)
        Fg = Fg.view(B, T, *Fg.shape[1:])
        Ft = Ft.view(B, T, *Ft.shape[1:])
        _, _, Cg, Hg, Wg = Fg.shape
        _, _, Ct, Ht, Wt = Ft.shape
        Fg_align = F.interpolate(
            Fg.view(B * T, Cg, Hg, Wg),
            size=(Ht, Wt),
            mode='bilinear',
            align_corners=False,
        ).view(B, T, Cg, Ht, Wt)
        A_st = self.gspm(Fg_align)
        F_tex_enh = self.gsem(Ft, A_st)
        Z = self.gsfm(F_tex_enh, Fg_align, A_st)
        Z_map = Z.permute(0, 2, 1).unsqueeze(-1)
        A_temporal = A_st.mean(dim=(-2, -1)).view(B, T, 1, 1).permute(0, 2, 1, 3)
        tokens = self.gsam(Z_map, attention_map=A_temporal)
        proj = self.projector(tokens)
        return proj.mean(dim=1)

    def compute_loss(self, x, labels, epoch=None, logits=None, frame_labels=None,
                     inconsistency_labels=None, lambda_inc=1.0, lambda_transition=0.5,
                     margin_transition=1.0, use_inc=True):
        """
        Dual-head loss:
          L_total = L_frame + lambda1 * L_incons + lambda2 * L_transition
        where:
          L_frame: frame-level source BCE
          L_incons: inconsistency BCE
          L_transition: margin constraint on state differences
        """
        if self.use_dual_head and frame_labels is not None:
            out = self.forward(x, apply_sigmoid=False)
            video_logits, frame_logits, inc_logits, delta_norm = out
            # Clamp logits to avoid numerical overflow inside BCE (exp -> NaN).
            frame_logits = torch.clamp(frame_logits, -50.0, 50.0)
            inc_logits = torch.clamp(inc_logits, -50.0, 50.0)
            L_frame = F.binary_cross_entropy_with_logits(frame_logits, frame_labels)
            loss_dict = {'L_frame': L_frame.item(), 'total_loss': L_frame.item()}
            total = L_frame

            if use_inc and inconsistency_labels is not None and lambda_inc > 0:
                L_incons = F.binary_cross_entropy_with_logits(
                    inc_logits.squeeze(-1), inconsistency_labels
                )
                total = total + lambda_inc * L_incons
                loss_dict['L_incons'] = L_incons.item()

            if (lambda_transition > 0 and inconsistency_labels is not None and
                    delta_norm is not None):
                delta_norm = torch.clamp(delta_norm, min=0.0, max=5.0)
                margin = margin_transition
                z = inconsistency_labels
                pos = z * F.relu(margin - delta_norm)
                neg = (1 - z) * delta_norm
                L_transition = (pos + neg).mean()
                total = total + lambda_transition * L_transition
                loss_dict['L_transition'] = L_transition.item()

            loss_dict['total_loss'] = total.item()
            return total, loss_dict
        # Single-head: video-level BCE (compatible with multi-label setup).
        if logits is None:
            logits = self.forward(x, apply_sigmoid=False)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if labels.dim() == 3:
            labels = labels.mean(dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, {'bce_loss': loss.item(), 'total_loss': loss.item()}
