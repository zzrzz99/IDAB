"""
IDABTemporalSTTPV1 主模型（旧版 v1，已废弃）
整合所有模块：EfficientNet + 纹理增强 + 帧间注意力重权 + STTP 池化 + MAA + ML-FFormer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientnet_encoder_v1 import EfficientNetB4V1
from .texture_enhance_v1 import TextureEnhanceBlockV1
from .temporal_diff_branch_v1 import FrameAttentionV1
from .sttp_pooling_v1 import SpatioTemporalTransformerPoolingV1
from .maa_module_v1 import TokenAttentionAggregatorV1
from .mlfformer_v1 import MLFFormerV1


class IDABTemporalSTTPV1(nn.Module):
    def __init__(self, num_classes=5, embed_size=256, num_heads=8, num_layers=2,
                 num_tokens=5, dim=512, sttp_heads=4, sttp_depth=2,
                 dropout: float = 0.3, sttp_dropout: float = 0.2,
                 transformer_dropout: float = 0.3, use_prompt=True):
        """
        Args:
            num_classes: 分类类别数
            embed_size: 嵌入维度
            num_heads: Transformer 注意力头数
            num_layers: Transformer 层数
            num_tokens: Token 聚合器 token 数量
            dim: 特征融合维度
            sttp_heads: STTP Transformer 注意力头数
            sttp_depth: STTP Transformer 深度
            use_prompt: 是否使用提示词学习（保留接口，当前版本暂不使用）
        """
        super(IDABTemporalSTTPV1, self).__init__()

        # 输入预处理
        self.input_conv = nn.Conv2d(3, 48, kernel_size=3, padding=1, stride=2)

        # EfficientNet backbone
        self.efficientnet = EfficientNetB4V1()

        # 纹理增强分支
        self.texture_block = TextureEnhanceBlockV1(32)

        # 帧间注意力重权模块（使用 L2 特征）
        self.frame_attention = FrameAttentionV1(in_channels=32, reduction=4)

        # 全局特征投影（L5 特征通过 GAP 后投影到统一维度）
        self.global_proj = nn.Linear(160, dim)  # L5 特征 GAP 后投影
        self.global_att_gate = nn.Linear(dim, 1)

        # 时空 Transformer 池化
        self.sttp = SpatioTemporalTransformerPoolingV1(
            dim=dim,
            num_heads=sttp_heads,
            depth=sttp_depth,
            dropout=sttp_dropout
        )

        # 多注意力聚合器（MAA）- 接收纹理+全局增强特征
        self.maa = TokenAttentionAggregatorV1(in_dim=32 + 160, num_tokens=num_tokens)

        # 最终投影和分类
        self.projector = nn.Sequential(
            nn.LayerNorm(num_tokens * (32 + 160)),
            nn.Dropout(dropout),
            nn.Linear(num_tokens * (32 + 160), embed_size),
            nn.Dropout(dropout)
        )
        self.mlfformer = MLFFormerV1(
            embed_size=embed_size,
            num_heads=num_heads,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=transformer_dropout
        )

        self.num_tokens = num_tokens
        self.dim = dim
        self.embed_size = embed_size

    def forward(self, x, return_frame_predictions=False, return_attention=False, apply_sigmoid=True):
        """
        Args:
            x: (B, T, C, H, W) 输入视频帧序列
            return_frame_predictions: 是否返回帧级预测
            return_attention: 是否返回注意力图（用于可视化）
        Returns:
            如果 return_frame_predictions=False and return_attention=False:
                output: (B, num_classes) 视频级预测
            如果 return_frame_predictions=True:
                video_pred: (B, num_classes) 视频级预测
                frame_predictions: (B, T, num_classes) 帧级预测
            如果 return_attention=True:
                还会返回 attention_maps 列表
        """
        B, T, C, H, W = x.shape

        # 输入预处理
        x_processed = self.input_conv(x.view(-1, C, H, W))  # (B*T, 48, H//2, W//2)

        # 提取全局特征（L2 和 L5）
        l2_feat, l5_feat = self.efficientnet(x_processed)  # l2: (B*T, 32, ...), l5: (B*T, 160, ...)

        # 纹理增强（使用 L2 特征）
        texture_feat = self.texture_block(l2_feat)  # (B*T, 32, ...)

        # 重塑为批次×时间维度
        l2_feat_reshaped = l2_feat.view(B, T, *l2_feat.shape[1:])  # (B, T, 32, H2, W2)
        l5_feat_reshaped = l5_feat.view(B, T, *l5_feat.shape[1:])  # (B, T, 160, H5, W5)
        texture_feat_reshaped = texture_feat.view(B, T, *texture_feat.shape[1:])  # (B, T, 32, H2, W2)

        # 将 L5 特征上采样到 L2 分辨率以便融合
        H2, W2 = l2_feat_reshaped.shape[-2:]
        l5_resized = F.interpolate(
            l5_feat_reshaped.view(B * T, *l5_feat.shape[1:]),
            size=(H2, W2),
            mode='bilinear',
            align_corners=False,
        ).view(B, T, 160, H2, W2)

        # 处理帧间注意力重权
        weighted_global_vectors = []
        attention_maps = []

        for t in range(T - 1):
            # 获取相邻帧的 L2 特征
            F_curr_l2 = l2_feat_reshaped[:, t]  # (B, 32, H2, W2)
            F_next_l2 = l2_feat_reshaped[:, t + 1]  # (B, 32, H2, W2)

            # 获取当前帧的 L5 全局特征（GAP）
            l5_t = l5_feat_reshaped[:, t]  # (B, 160, H5, W5)
            G_curr = F.adaptive_avg_pool2d(l5_t, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 160)

            # 投影到统一维度
            G_curr_proj = self.global_proj(G_curr)  # (B, dim)

            # 帧间注意力重权
            _, att = self.frame_attention(F_curr_l2, F_next_l2, G_curr_proj)

            l5_resized_t = l5_resized[:, t]  # (B, 160, H2, W2)
            att_expanded = att
            weighted_map = l5_resized_t * att_expanded  # (B, 160, H2, W2)
            weighted_vec = F.adaptive_avg_pool2d(weighted_map, (1, 1)).flatten(1)  # (B, 160)
            weighted_proj = self.global_proj(weighted_vec)  # (B, dim)

            weighted_global_vectors.append(weighted_proj)
            attention_maps.append(att)

        # 堆叠为序列以输入 STTP
        if len(weighted_global_vectors) > 0:
            fused_features = torch.stack(weighted_global_vectors, dim=1)  # (B, T-1, dim)
            attention_stack = torch.stack(attention_maps, dim=1)  # (B, T-1, 1, H2, W2)
        else:
            # 如果没有帧间差分（T=1），使用全局特征
            l5_t = l5_feat_reshaped[:, 0]  # (B, 160, H5, W5)
            G_curr = F.adaptive_avg_pool2d(l5_t, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 160)
            G_curr_proj = self.global_proj(G_curr)  # (B, dim)
            fused_features = G_curr_proj.unsqueeze(1)  # (B, 1, dim)
            attention_stack = torch.ones(B, 1, 1, H2, W2, device=x.device, dtype=x.dtype)

        # 时空 Transformer 池化 -> 全局表征
        pooled = self.sttp(fused_features)  # (B, dim)

        # 生成全局注意力图（融合时间维度）
        global_att_gate = torch.sigmoid(self.global_att_gate(pooled)).view(B, 1, 1, 1)
        global_attention_map = torch.clamp(attention_stack.mean(dim=1) * global_att_gate, min=0.0, max=1.0)

        # 纹理与全局特征的空间增强
        texture_mean = texture_feat_reshaped.mean(dim=1)  # (B, 32, H2, W2)
        global_mean = l5_resized.mean(dim=1)  # (B, 160, H2, W2)

        enhanced_texture = texture_mean * global_attention_map  # (B, 32, H2, W2)
        enhanced_global = global_mean * global_attention_map  # (B, 160, H2, W2)

        fused_spatial = torch.cat([enhanced_texture, enhanced_global], dim=1)  # (B, 32+160, H2, W2)

        # 多注意力聚合（MAA），在全局注意力图引导下生成 tokens
        tokens = self.maa(fused_spatial, attention_map=global_attention_map)  # (B, num_tokens, 192)

        tokens_flat = tokens.view(B, -1)  # (B, num_tokens*192)
        proj = self.projector(tokens_flat)  # (B, embed_size)
        proj = proj.unsqueeze(1)  # (B, 1, embed_size)

        # ML-FFormer 分类
        if return_frame_predictions:
            # 帧级预测：对每帧单独预测
            frame_predictions = []
            for t in range(T):
                if t < attention_stack.shape[1]:
                    frame_att = attention_stack[:, t]
                else:
                    frame_att = global_attention_map
                frame_texture = texture_feat_reshaped[:, t]
                frame_global = l5_resized[:, t]
                frame_enhanced = torch.cat(
                    [frame_texture * frame_att, frame_global * frame_att],
                    dim=1,
                )
                frame_tokens = self.maa(frame_enhanced, attention_map=frame_att)
                frame_tokens_flat = frame_tokens.view(B, -1)
                frame_proj = self.projector(frame_tokens_flat).unsqueeze(1)
                frame_logits = self.mlfformer(frame_proj)
                frame_out = torch.sigmoid(frame_logits) if apply_sigmoid else frame_logits
                frame_predictions.append(frame_out)
            frame_predictions = torch.stack(frame_predictions, dim=1)  # (B, T, num_classes)

            # 视频级预测
            video_logits = self.mlfformer(proj)  # (B, num_classes)
            video_out = torch.sigmoid(video_logits) if apply_sigmoid else video_logits

            if return_attention:
                return video_out, frame_predictions, attention_stack
            return video_out, frame_predictions
        else:
            # 标准模式：只返回视频级预测
            logits = self.mlfformer(proj)
            out = torch.sigmoid(logits) if apply_sigmoid else logits
            if return_attention:
                return out, attention_stack
            return out

    def compute_loss(self, x, labels, epoch=None, logits=None, frame_labels=None):
        """
        计算损失函数
        Args:
            x: (B, T, C, H, W) 输入视频
            labels: (B, num_classes) 或 (B, T, num_classes) 标签
            epoch: 当前 epoch（可选）
        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        if logits is None:
            logits = self.forward(x, apply_sigmoid=False)

        # 确保标签形状匹配
        if labels.dim() == 3:  # (B, T, num_classes)
            labels = labels.mean(dim=1)  # (B, num_classes)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, {'bce_loss': loss.item(), 'total_loss': loss.item()}
