# IDAB: Inconsistency Detection with Adaptive Bifurcation

## 项目概述

IDAB 是一个面向帧级伪造来源分类与帧间不一致性检测的深度伪造检测框架。主要核心模块包括：

1. **GSPM（Generation State Perception Module）**：通过帧间特征差分感知生成状态，生成时空状态注意力图 A_st
2. **GSEM（Generation State Enhancement Module）**：使用状态注意力增强纹理特征
3. **GSFM（Generation State Fusion Module）**：注意力引导的双线性池化融合纹理与全局特征
4. **GSAM（Generation State Aggregation Module）**：K 分支注意力聚合生成 tokens
5. **GS-Former**：基于 Transformer 的生成状态编码器，输出视频级预测

## 模型架构

```
                ┌──────────────────────┐
                │   EfficientNet-B4    │
                ├──────────────────────┤
        L2 ──► Texture Features ──► F_tex ─┐
        L5 ──► Global Features  ──► F_glb ─┤
                     │
              GSPM (Frame Difference + State Attention)
                     ↓
              GSEM (State-guided Texture Enhancement)
                     ↓
              GSFM (Attention-guided Bilinear Fusion)
                     ↓
              GSAM (Generation State Aggregation)
                     ↓
              GS-Former (Transformer Encoder)
                     ↓
              Multi-label Classification
```

## 目录结构

```
IDAB/
│
├── configs/
│   └── idab.yaml               # IDAB 模型与训练超参
│
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── idab.py                 # 稳定导入路径（re-export）
│   ├── idab_v2.py              # IDAB 主模型
│   ├── backbone/
│   │   └── efficientnet.py     # EfficientNetBackbone
│   ├── perception/             # GSPM
│   │   ├── frame_difference.py
│   │   ├── state_attention.py
│   │   └── gspm.py
│   ├── gsem/
│   │   └── gsem.py
│   ├── gsfm/
│   │   └── gsfm.py
│   ├── gsam/
│   │   └── gsam.py
│   ├── gs_former/
│   │   ├── gs_former_encoder.py
│   │   └── gs_former.py
│   ├── idab_temporal_sttp_v1.py  # 旧版 v1（已废弃）
│   ├── mlfformer_v1.py           # 旧版 v1（已废弃）
│   ├── transformer_v1.py         # 旧版 v1（已废弃）
│   ├── maa_module_v1.py          # 旧版 v1（已废弃）
│   ├── texture_enhance_v1.py     # 旧版 v1（已废弃）
│   ├── efficientnet_encoder_v1.py # 旧版 v1（已废弃）
│   ├── sttp_pooling_v1.py        # 旧版 v1（已废弃）
│   ├── temporal_diff_branch_v1.py # 旧版 v1（已废弃）
│   └── cross_attention_fusion_v1.py # 旧版 v1（已废弃）
│
├── utils/
│   ├── dataset_loader.py
│   ├── transforms.py
│   ├── metrics.py
│   ├── trainer_core.py
│   ├── train_utils.py
│   └── misc.py
│
├── data/
│   ├── datasets.py             # 视频帧加载
│   ├── transforms.py
│   ├── preprocess.py
│   └── ...                     # FH-FF++ 构建脚本
│
├── train.py                     # 支持 --model idab --config configs/idab.yaml
├── eval.py                      # IDAB 评估
├── test.py
├── requirements.txt
└── README.md
```

## 训练 IDAB

```bash
# 使用配置文件，前 5 个 epoch 冻结 backbone
python train.py --model idab --config configs/idab.yaml --data_dir /path/to/dataset --freeze_backbone_epochs 5

# 仅指定模型，其余命令行参数
python train.py --model idab --data_dir /path/to/dataset --dropout 0.5 --epochs 60
```

## 评估 IDAB

```bash
python eval.py --model idab --weights checkpoints/best_model.pth --data_dir /path/to/dataset --config configs/idab.yaml
```

## 环境安装

```bash
# 创建 conda 环境
conda create -n idab python=3.10
conda activate idab

# 安装依赖
pip install -r requirements.txt
```

## 数据预处理

如果您的数据集是视频文件，需要先提取帧：

```bash
# 处理所有数据集分割
python data/preprocess.py --dataset /path/to/dataset --split all

# 或单独处理某个分割
python data/preprocess.py --dataset /path/to/dataset --split train
```

数据集目录结构应该如下：

```
dataset/
├── train/
│   ├── label1/
│   │   ├── video1/
│   │   │   ├── frame_000000.jpg
│   │   │   ├── frame_000001.jpg
│   │   │   └── ...
│   │   └── video2/
│   │       └── ...
│   └── label2/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## 训练

### 使用默认参数训练

```bash
python train.py --data_dir /path/to/dataset
```

### 使用自定义参数训练

```bash
python train.py \
    --data_dir /path/to/dataset \
    --frames_per_video 16 \
    --batch_size 2 \
    --epochs 60 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --save_dir checkpoints \
    --device cuda
```

### 训练参数说明

- `--data_dir`: 数据集根目录
- `--frames_per_video`: 每个视频使用的帧数（默认 16）
- `--batch_size`: 批次大小（默认 4）
- `--epochs`: 训练轮数（默认 40）
- `--lr`: 学习率（默认 3e-4）
- `--weight_decay`: 权重衰减（默认 5e-4）
- `--save_dir`: 模型保存目录（默认 checkpoints）
- `--device`: 设备（默认 cuda，如果可用）
- `--acc_type`: 准确率类型（all/per_label/mean，默认 all）

### 多卡分布式训练（DDP）

`train.py` 已内置 PyTorch DistributedDataParallel 支持，可使用 `torchrun` 在多张 GPU 上训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 train.py \
  --data_dir /root/autodl-tmp/datasets \
  --batch_size 8 \
  --frames_per_video 12
```

说明：

- `torchrun` 会自动设置 `RANK/LOCAL_RANK` 环境变量，脚本会据此初始化 NCCL 通信。
- 建议 `--batch_size` 为单卡 batch（上例表示每卡 8，总有效 batch 为 16）。
- 训练与验证损失、准确率会在各卡汇总后仅在主进程（rank0）打印与保存，最终测试同样只在主进程执行。

## 测试

```bash
python test.py \
    --data_dir /path/to/dataset \
    --weights checkpoints/best_model.pth \
    --frames_per_video 16 \
    --batch_size 2
```

### 测试参数说明

- `--data_dir`: 数据集根目录
- `--weights`: 模型权重路径（必需）
- `--frames_per_video`: 每个视频使用的帧数（默认 8）
- `--batch_size`: 批次大小（默认 4）
- `--device`: 设备（默认 cuda，如果可用）
- `--acc_type`: 准确率类型（all/per_label/mean，默认 all）

## 模型配置

模型的主要超参数可以在 `models/idab_v2.py` 或 `configs/idab.yaml` 中调整：

- `num_classes`: 分类类别数（会根据数据集自动调整）
- `embed_size`: 嵌入维度（默认 256）
- `num_heads`: Transformer 注意力头数（默认 8）
- `num_layers`: Transformer 层数（默认 2）
- `num_tokens`: GSAM token 数量（默认 5）
- `token_dim`: token 特征维度（默认 512）
- `bilinear_d`: GSFM 投影维度（默认 32）

## 输出结果

训练和测试完成后，会生成以下文件：

1. `train_test_results.json`: 包含训练历史和测试指标
2. `frame_predictions_results.json`: 包含每个视频的帧级预测结果
3. `checkpoints/best_model.pth`: 最佳模型权重
4. `checkpoints/checkpoint_epoch*.pth`: 定期保存的检查点

## 评估指标

模型支持以下评估指标：

- **Accuracy**: 全标签全对准确率
- **Per-label Accuracy**: 逐标签准确率
- **Mean Accuracy**: 单标签平均准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1 分数
- **AUC**: ROC-AUC（如果可用）
- **Frame-wise F1-score**: 每帧伪造识别能力
- **Inconsistency AUC/F1**: 帧间不一致性检测指标

## 注意事项

1. 模型需要至少 2 个连续帧才能计算帧间差分
2. 如果视频只有 1 帧，GSPM 将使用首帧复制策略生成注意力图
3. 建议使用 GPU 训练，单卡 24GB 显存可以支持 batch_size=4 的训练
4. 旧版 v1 文件（`*_v1.py`）已废弃，仅保留用于兼容，不建议用于新实验

## 引用

如果您使用了本模型，请引用：

```bibtex
@article{idab2026,
  title={IDAB: Inconsistency Detection with Adaptive Bifurcation for Frame-wise Heterogeneous Deepfake Detection},
  author={Your Name},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026}
}
```

## 许可证

本项目遵循原项目许可证。

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
