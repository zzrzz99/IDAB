# 旧版 v1（已废弃，保留兼容；无 efficientnet_pytorch 时跳过）
try:
    from .idab_temporal_sttp_v1 import IDABTemporalSTTPV1
    from .temporal_diff_branch_v1 import FrameAttentionV1, TemporalDiffBranchV1
    from .cross_attention_fusion_v1 import CrossAttentionFusionV1
    from .sttp_pooling_v1 import SpatioTemporalTransformerPoolingV1
    _V1_AVAILABLE = True
except ImportError:
    IDABTemporalSTTPV1 = None
    FrameAttentionV1 = None
    TemporalDiffBranchV1 = None
    CrossAttentionFusionV1 = None
    SpatioTemporalTransformerPoolingV1 = None
    _V1_AVAILABLE = False

from .idab import IDAB

# 新模块（论文最新命名）
from .perception import GSPM, InterFrameDifference, StateAttentionGenerator
from .gsem import GSEM
from .gsfm import GSFM
from .gsam import GSAM
from .gs_former import GSFormer, GSFormerEncoder

__all__ = [
    # 主模型
    'IDAB',
    # 新模块
    'GSPM',
    'GSEM',
    'GSFM',
    'GSAM',
    'GSFormer',
    'GSFormerEncoder',
    # 兼容性别名
    'InterFrameDifference',
    'StateAttentionGenerator',
]

if _V1_AVAILABLE:
    __all__ += [
        'IDABTemporalSTTPV1',
        'FrameAttentionV1',
        'TemporalDiffBranchV1',
        'CrossAttentionFusionV1',
        'SpatioTemporalTransformerPoolingV1',
    ]
