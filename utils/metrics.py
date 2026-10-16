"""
评估指标
从原始 ETMA 复制并适配（旧版术语，功能不变）
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import json

try:
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def evaluate(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    per_label_acc = (all_labels == all_preds).float().mean(dim=0).tolist()
    mean_acc = (all_labels == all_preds).float().mean().item()
    try:
        auc = roc_auc_score(all_labels, all_probs, average='macro')
    except Exception:
        auc = None
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1,
            'per_label_accuracy': per_label_acc, 'mean_accuracy': mean_acc, 'auc': auc}


def _optimize_threshold_per_class(y_true, y_probs, label_names):
    """
    为每个类别优化阈值，使用F1-score作为优化目标
    返回: 每个类别的最优阈值
    """
    optimal_thresholds = {}
    for idx, label_name in enumerate(label_names):
        y_true_class = y_true[:, idx]
        y_probs_class = y_probs[:, idx]
        
        # 只优化有正样本的类别
        if y_true_class.sum() == 0:
            optimal_thresholds[label_name] = 0.5
            continue
        
        def f1_at_threshold(thresh):
            y_pred = (y_probs_class >= thresh).astype(float)
            if y_pred.sum() == 0:
                return 0.0
            tp = ((y_pred == 1) & (y_true_class == 1)).sum()
            fp = ((y_pred == 1) & (y_true_class == 0)).sum()
            fn = ((y_pred == 0) & (y_true_class == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return -f1  # 最小化负F1 = 最大化F1
        
        if HAS_SCIPY:
            # 使用scipy优化
            result = minimize_scalar(f1_at_threshold, bounds=(0.01, 0.99), method='bounded')
            optimal_thresholds[label_name] = result.x
        else:
            # Fallback: 网格搜索
            best_thresh = 0.5
            best_f1 = -f1_at_threshold(0.5)
            for thresh in np.linspace(0.01, 0.99, 99):
                f1_val = -f1_at_threshold(thresh)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_thresh = thresh
            optimal_thresholds[label_name] = best_thresh
    
    return optimal_thresholds


def _optimize_threshold_for_eacc(y_true, y_probs, label_names, f1_weight=0.5, eacc_weight=0.5):
    """
    优化阈值，同时考虑F1-score和EACC（整体准确率）
    使用联合优化策略：先优化F1，然后在保持F1不显著下降的前提下优化EACC
    
    Args:
        y_true: 真实标签 (n_samples, n_classes)
        y_probs: 预测概率 (n_samples, n_classes)
        label_names: 标签名称列表
        f1_weight: F1权重（默认0.5）
        eacc_weight: EACC权重（默认0.5）
    
    Returns:
        每个类别的最优阈值
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    # 首先使用F1优化得到初始阈值
    f1_thresholds = _optimize_threshold_per_class(y_true, y_probs, label_names)
    
    # 计算初始F1和EACC
    initial_preds = np.zeros_like(y_probs)
    for idx, label_name in enumerate(label_names):
        initial_preds[:, idx] = (y_probs[:, idx] >= f1_thresholds[label_name]).astype(float)
    initial_f1 = f1_score(y_true, initial_preds, average='samples', zero_division=0)
    initial_eacc = accuracy_score(y_true, initial_preds)
    
    print(f'  初始F1优化结果: F1={initial_f1:.4f}, EACC={initial_eacc:.4f}')
    
    # 联合优化：在F1阈值附近微调以提升EACC
    # 使用全局搜索策略：同时优化所有类别的阈值
    optimal_thresholds = f1_thresholds.copy()
    
    # 简化策略：对每个类别，在F1最优阈值附近微调，选择能提升EACC的阈值
    best_eacc = initial_eacc
    best_f1 = initial_f1
    
    # 对每个类别进行微调
    for idx, label_name in enumerate(label_names):
        y_true_class = y_true[:, idx]
        y_probs_class = y_probs[:, idx]
        
        if y_true_class.sum() == 0:
            optimal_thresholds[label_name] = 0.5
            continue
        
        base_thresh = f1_thresholds[label_name]
        best_thresh = base_thresh
        
        # 在base_thresh附近搜索（±0.15范围）
        search_range = (max(0.01, base_thresh - 0.15), min(0.99, base_thresh + 0.15))
        search_points = np.linspace(search_range[0], search_range[1], 30)
        
        for thresh in search_points:
            # 使用当前阈值预测
            y_pred_class = (y_probs_class >= thresh).astype(float)
            
            # 计算F1
            tp = ((y_pred_class == 1) & (y_true_class == 1)).sum()
            fp = ((y_pred_class == 1) & (y_true_class == 0)).sum()
            fn = ((y_pred_class == 0) & (y_true_class == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 计算EACC（需要所有类别的预测）
            temp_preds = initial_preds.copy()
            temp_preds[:, idx] = y_pred_class
            eacc = accuracy_score(y_true, temp_preds)
            
            # 联合得分：F1和EACC的加权平均
            # 要求F1不能下降太多（至少保持初始F1的90%）
            if f1 >= initial_f1 * 0.90:
                score = f1_weight * f1 + eacc_weight * eacc
                # 如果EACC提升且F1不显著下降，更新阈值
                if eacc > best_eacc or (eacc >= best_eacc * 0.98 and f1 > best_f1 * 0.95):
                    best_thresh = thresh
                    best_eacc = eacc
                    best_f1 = f1
        
        optimal_thresholds[label_name] = best_thresh
        # 更新initial_preds以便下一个类别使用
        initial_preds[:, idx] = (y_probs_class >= best_thresh).astype(float)
    
    # 重新计算最终指标
    final_preds = np.zeros_like(y_probs)
    for idx, label_name in enumerate(label_names):
        final_preds[:, idx] = (y_probs[:, idx] >= optimal_thresholds[label_name]).astype(float)
    final_f1 = f1_score(y_true, final_preds, average='samples', zero_division=0)
    final_eacc = accuracy_score(y_true, final_preds)
    print(f'  最终优化结果: F1={final_f1:.4f}, EACC={final_eacc:.4f}')
    
    return optimal_thresholds


def evaluate_with_frame_predictions(model, test_loader, device, label_names, threshold=0.5, epoch=None, optimize_threshold=False, optimize_for_eacc=False):
    """
    评估模型并输出帧级预测结果
    """
    model.eval()
    all_video_labels = []
    all_video_preds = []
    all_video_probs = []
    frame_predictions_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 3:
                images, labels, video_names = batch
            else:
                images, labels = batch
                video_names = [f"video_{batch_idx}_{i}" for i in range(images.size(0))]
            images = images.to(device)
            if epoch is not None and hasattr(model, 'forward') and 'epoch' in model.forward.__code__.co_varnames:
                video_probs, frame_preds = model(images, return_frame_predictions=True, epoch=epoch)
            else:
                video_probs, frame_preds = model(images, return_frame_predictions=True)

            # 视频级预测
            video_probs_np = video_probs.cpu().numpy()
            video_preds = (video_probs > threshold).cpu().numpy()
            all_video_probs.extend(video_probs_np)
            all_video_preds.extend(video_preds)
            all_video_labels.extend(labels.numpy())

            # 帧级预测处理
            frame_preds = frame_preds.cpu().numpy()  # (batch, frames, num_classes)
            batch_size = images.size(0)
            for i in range(batch_size):
                video_result = {
                    'video_id': video_names[i],
                    'video_level_predictions': {},
                    'frame_level_predictions': {}
                }

                # 视频级预测结果
                for j, label_name in enumerate(label_names):
                    video_result['video_level_predictions'][label_name] = {
                        'predicted': bool(video_preds[i][j]),
                        'confidence': float(video_probs_np[i][j])
                    }

                # 帧级预测结果
                for j, label_name in enumerate(label_names):
                    frame_indices = []
                    frame_confidences = []

                    for frame_idx in range(frame_preds.shape[1]):
                        if frame_preds[i, frame_idx, j] > threshold:
                            frame_indices.append(int(frame_idx))
                            frame_confidences.append(float(frame_preds[i, frame_idx, j]))

                    video_result['frame_level_predictions'][label_name] = {
                        'detected_frames': frame_indices,
                        'frame_confidences': frame_confidences,
                        'total_detected_frames': len(frame_indices)
                    }

                frame_predictions_results.append(video_result)

    # 计算视频级指标
    # 确保所有数组形状一致，避免tensor转换错误
    all_video_preds = np.array(all_video_preds)
    all_video_labels = np.array(all_video_labels)
    all_video_probs = np.array(all_video_probs)
    
    # 验证形状
    if len(all_video_preds.shape) != 2:
        raise ValueError(f"all_video_preds shape error: {all_video_preds.shape}, expected (N, num_classes)")
    if len(all_video_labels.shape) != 2:
        raise ValueError(f"all_video_labels shape error: {all_video_labels.shape}, expected (N, num_classes)")
    if all_video_preds.shape != all_video_labels.shape:
        raise ValueError(f"Shape mismatch: preds {all_video_preds.shape} vs labels {all_video_labels.shape}")
    
    # 🔍 诊断信息：检查预测分布和概率分布
    num_samples, num_classes = all_video_preds.shape
    pred_per_class = all_video_preds.sum(axis=0)
    label_per_class = all_video_labels.sum(axis=0)
    
    print(f'\n🔍 预测分布诊断:')
    print(f'  样本数: {num_samples}, 类别数: {num_classes}')
    print(f'  预测分布 (每类预测为正的样本数): {pred_per_class}')
    print(f'  标签分布 (每类实际为正的样本数): {label_per_class}')
    
    # 检查每个类别的概率分布
    print(f'\n🔍 各类别概率分布统计 (阈值=0.5):')
    for idx, label_name in enumerate(label_names):
        class_probs = all_video_probs[:, idx]
        class_labels = all_video_labels[:, idx]
        class_preds = all_video_preds[:, idx]
        
        pos_probs = class_probs[class_labels == 1]  # 正样本的概率
        neg_probs = class_probs[class_labels == 0]  # 负样本的概率
        
        print(f'  {label_name}:')
        print(f'    正样本数: {int(label_per_class[idx])}, 预测为正数: {int(pred_per_class[idx])}')
        if len(pos_probs) > 0:
            print(f'    正样本概率范围: [{pos_probs.min():.3f}, {pos_probs.max():.3f}], 平均: {pos_probs.mean():.3f}, 中位数: {np.median(pos_probs):.3f}')
        if len(neg_probs) > 0:
            print(f'    负样本概率范围: [{neg_probs.min():.3f}, {neg_probs.max():.3f}], 平均: {neg_probs.mean():.3f}, 中位数: {np.median(neg_probs):.3f}')
        
        if pred_per_class[idx] == 0 and label_per_class[idx] > 0:
            print(f'    ⚠️  严重警告：该类别预测全为0，但实际有{int(label_per_class[idx])}个正样本！')
            if len(pos_probs) > 0:
                optimal_thresh = np.percentile(pos_probs, 10)  # 使用10%分位数作为建议阈值
                print(f'    建议阈值: {optimal_thresh:.3f} (当前0.5过高)')
        elif pred_per_class[idx] < label_per_class[idx] * 0.5:
            print(f'    ⚠️  警告：预测数量远少于实际数量，阈值可能过高')
    
    if np.any(pred_per_class == 0) or np.any(pred_per_class == num_samples):
        print(f'\n  ⚠️  警告：某些类别预测全为0或全为1，这是类别不平衡+固定阈值问题！')
        print(f'  建议：使用类别特定的阈值优化（见下方优化阈值结果）')
    
    # 使用固定阈值0.5的指标
    acc = accuracy_score(all_video_labels, all_video_preds)
    precision = precision_score(all_video_labels, all_video_preds, average='samples', zero_division=0)
    recall = recall_score(all_video_labels, all_video_preds, average='samples', zero_division=0)
    f1 = f1_score(all_video_labels, all_video_preds, average='samples', zero_division=0)
    # 使用numpy计算，避免tensor转换问题
    all_video_labels_t = torch.tensor(all_video_labels, dtype=torch.float32)
    all_video_preds_t = torch.tensor(all_video_preds, dtype=torch.float32)
    per_label_acc = (all_video_labels_t == all_video_preds_t).float().mean(dim=0).tolist()
    mean_acc = (all_video_labels_t == all_video_preds_t).float().mean().item()
    try:
        auc = roc_auc_score(all_video_labels, all_video_probs, average='macro')
    except Exception:
        auc = None

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_label_accuracy': per_label_acc,
        'mean_accuracy': mean_acc,
        'auc': auc
    }
    
    # 如果启用阈值优化，计算优化后的指标
    print(f'\n[DEBUG] evaluate_with_frame_predictions: optimize_threshold={optimize_threshold} (type: {type(optimize_threshold)})')
    if optimize_threshold:
        if not HAS_SCIPY:
            print(f'\n⚠️  警告：scipy未安装，将使用网格搜索（较慢）')
        
        if optimize_for_eacc:
            print(f'\n🔧 联合优化阈值（考虑F1和EACC）:')
            optimal_thresholds = _optimize_threshold_for_eacc(
                np.array(all_video_labels), 
                np.array(all_video_probs), 
                label_names,
                f1_weight=0.4,  # 稍微降低F1权重，提升EACC
                eacc_weight=0.6
            )
        else:
            print(f'\n🔧 类别特定阈值优化（仅F1）:')
            optimal_thresholds = _optimize_threshold_per_class(
                np.array(all_video_labels), 
                np.array(all_video_probs), 
                label_names
            )
        
        # 使用优化后的阈值重新预测
        optimized_preds = np.zeros_like(all_video_preds)
        for idx, label_name in enumerate(label_names):
            thresh = optimal_thresholds[label_name]
            optimized_preds[:, idx] = (all_video_probs[:, idx] >= thresh).astype(float)
            print(f'  {label_name}: 阈值={thresh:.3f}')
        
        # 计算优化后的指标
        opt_acc = accuracy_score(all_video_labels, optimized_preds)
        opt_precision = precision_score(all_video_labels, optimized_preds, average='samples', zero_division=0)
        opt_recall = recall_score(all_video_labels, optimized_preds, average='samples', zero_division=0)
        opt_f1 = f1_score(all_video_labels, optimized_preds, average='samples', zero_division=0)
        
        opt_pred_per_class = optimized_preds.sum(axis=0)
        print(f'\n📊 优化阈值后的预测分布:')
        print(f'  预测分布: {opt_pred_per_class}')
        print(f'  标签分布: {label_per_class}')
        
        print(f'\n📈 优化阈值后的指标:')
        print(f'  Accuracy: {opt_acc:.4f} (原: {acc:.4f})')
        print(f'  Precision: {opt_precision:.4f} (原: {precision:.4f})')
        print(f'  Recall: {opt_recall:.4f} (原: {recall:.4f})')
        print(f'  F1-Score: {opt_f1:.4f} (原: {f1:.4f})')
        
        metrics['optimized_thresholds'] = optimal_thresholds
        metrics['optimized_accuracy'] = opt_acc
        metrics['optimized_precision'] = opt_precision
        metrics['optimized_recall'] = opt_recall
        metrics['optimized_f1'] = opt_f1

    return {
        'video_level_metrics': metrics,
        'frame_level_predictions': frame_predictions_results
    }


def evaluate_inconsistency(model, test_loader, device, threshold=0.5, optimize_threshold=True):
    """
    不一致性任务评估：二分类（是否存在帧间生成机制切换）。
    报告视频级 AUC、AP、F1（可选优化阈值）、Accuracy；
    若模型返回帧级预测则汇总帧级 AUC。
    """
    model.eval()
    all_video_probs = []
    all_video_labels = []
    all_frame_probs = []
    all_frame_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                images, video_labels, frame_labels, _ = batch
            else:
                images, video_labels = batch[0], batch[1]
                frame_labels = None
            images = images.to(device)
            video_labels = video_labels.numpy()
            if frame_labels is not None:
                frame_labels = frame_labels.numpy()

            try:
                video_probs, frame_preds = model(images, return_frame_predictions=True)
            except TypeError:
                video_probs = model(images, apply_sigmoid=True)
                frame_preds = None
            video_probs = video_probs.cpu().numpy()
            if video_probs.ndim == 1:
                video_probs = video_probs[:, np.newaxis]
            all_video_probs.append(video_probs)
            all_video_labels.append(video_labels)

            if frame_preds is not None and frame_labels is not None:
                # frame_preds: (B, T, 1) 或 (B, T)
                fp = frame_preds.cpu().numpy()
                if fp.ndim == 3:
                    fp = fp.squeeze(-1)
                fl = frame_labels  # (B, T-1)
                for i in range(fp.shape[0]):
                    t = min(fp.shape[1], fl.shape[1])
                    all_frame_probs.extend(fp[i, :t].ravel().tolist())
                    all_frame_labels.extend(fl[i, :t].ravel().tolist())

    all_video_probs = np.concatenate(all_video_probs, axis=0).squeeze()
    all_video_labels = np.concatenate(all_video_labels, axis=0).squeeze()
    if all_video_probs.ndim == 0:
        all_video_probs = np.atleast_1d(all_video_probs)
    if all_video_labels.ndim == 0:
        all_video_labels = np.atleast_1d(all_video_labels)
    if all_video_probs.ndim == 1 and all_video_labels.ndim == 1:
        pass
    else:
        all_video_probs = all_video_probs.reshape(-1, 1)
        all_video_labels = all_video_labels.reshape(-1, 1).squeeze(axis=1)

    # 视频级指标（二分类）
    preds_05 = (all_video_probs >= threshold).astype(np.float64)
    acc = accuracy_score(all_video_labels, preds_05)
    precision = precision_score(all_video_labels, preds_05, zero_division=0)
    recall = recall_score(all_video_labels, preds_05, zero_division=0)
    f1 = f1_score(all_video_labels, preds_05, zero_division=0)
    try:
        auc = roc_auc_score(all_video_labels, all_video_probs)
    except ValueError:
        auc = float('nan')
    try:
        ap = average_precision_score(all_video_labels, all_video_probs)
    except ValueError:
        ap = float('nan')

    # 优化阈值（最大化 F1）
    opt_threshold = threshold
    if optimize_threshold and np.unique(all_video_labels).size >= 2 and HAS_SCIPY:
        def neg_f1(t):
            p = (all_video_probs >= t).astype(np.float64)
            return -f1_score(all_video_labels, p, zero_division=0)
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(neg_f1, bounds=(0.01, 0.99), method='bounded')
        opt_threshold = float(res.x)
        preds_opt = (all_video_probs >= opt_threshold).astype(np.float64)
        opt_f1 = f1_score(all_video_labels, preds_opt, zero_division=0)
        opt_acc = accuracy_score(all_video_labels, preds_opt)
    else:
        opt_f1 = f1
        opt_acc = acc

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ap': ap,
        'threshold': threshold,
        'optimized_threshold': opt_threshold,
        'optimized_f1': opt_f1,
        'optimized_accuracy': opt_acc,
    }

    # 帧级指标（若有）
    if all_frame_probs and all_frame_labels:
        all_frame_probs = np.array(all_frame_probs)
        all_frame_labels = np.array(all_frame_labels)
        if np.unique(all_frame_labels).size >= 2:
            try:
                frame_auc = roc_auc_score(all_frame_labels, all_frame_probs)
            except ValueError:
                frame_auc = float('nan')
        else:
            frame_auc = float('nan')
        metrics['frame_auc'] = frame_auc
        metrics['frame_num_samples'] = int(len(all_frame_labels))

    return {'video_level_metrics': metrics, 'frame_level_predictions': []}


def evaluate_dual_task(model, test_loader, device, label_names):
    """
    双头任务评估：主任务帧级来源分类 + 辅任务不一致性 + 视频级来源召回。
    对应论文：Table 1 帧级指标、Table 2 不一致性、Video-level Source Recall。
    test_loader 的 batch 为 dict: frames, frame_labels [B,T,C], inconsistency_labels [B,T-1].
    """
    model.eval()
    num_classes = len(label_names)
    all_frame_probs = []
    all_frame_labels = []
    all_inc_probs = []
    all_inc_labels = []
    # 视频级来源召回：每类 TP 与 support（视频数）
    video_tp = np.zeros(num_classes, dtype=np.float64)
    video_support = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for batch in test_loader:
            images = batch['frames'].to(device)
            fl = batch['frame_labels']  # (B, T, C)
            inc_lab = batch['inconsistency_labels']  # (B, T-1)
            out = model(images, apply_sigmoid=True)
            _, frame_logits, inc_logits = out[:3]
            frame_logits = frame_logits.cpu().numpy()
            inc_logits = inc_logits.cpu().numpy().squeeze(-1)
            fl = fl.numpy()
            inc_lab = inc_lab.numpy()
            B, T, C = frame_logits.shape
            all_frame_probs.append(frame_logits.reshape(-1, C))
            all_frame_labels.append(fl.reshape(-1, C))
            all_inc_probs.append(inc_logits.reshape(-1))
            all_inc_labels.append(inc_lab.reshape(-1))

            # 视频级来源召回：若视频中真实存在某类，只要任意一帧预测到即视为命中
            preds_b = (frame_logits >= 0.5).astype(np.float64)
            for c in range(C):
                video_has_c = (fl[:, :, c].sum(axis=1) > 0)
                video_pred_c = (preds_b[:, :, c].sum(axis=1) > 0)
                video_tp[c] += (video_has_c & video_pred_c).sum()
                video_support[c] += video_has_c.sum()

    all_frame_probs = np.concatenate(all_frame_probs, axis=0)
    all_frame_labels = np.concatenate(all_frame_labels, axis=0)
    all_inc_probs = np.concatenate(all_inc_probs, axis=0)
    all_inc_labels = np.concatenate(all_inc_labels, axis=0)

    # ---------- 表1：帧级伪造来源多标签分类（主任务） ----------
    preds = (all_frame_probs >= 0.5).astype(np.float64)
    frame_acc = accuracy_score(all_frame_labels, preds)
    frame_f1 = f1_score(all_frame_labels, preds, average='samples', zero_division=0)
    frame_f1_macro = f1_score(all_frame_labels, preds, average='macro', zero_division=0)
    per_class_f1 = {}
    for i, name in enumerate(label_names):
        try:
            per_class_f1[name] = float(f1_score(all_frame_labels[:, i], preds[:, i], zero_division=0))
        except Exception:
            per_class_f1[name] = 0.0
    try:
        frame_ap = average_precision_score(all_frame_labels, all_frame_probs, average='samples')
    except Exception:
        frame_ap = float('nan')

    # ---------- 表2：帧间不一致性检测（辅任务） ----------
    if np.unique(all_inc_labels).size >= 2:
        try:
            inc_auc = roc_auc_score(all_inc_labels, all_inc_probs)
        except ValueError:
            inc_auc = float('nan')
        inc_preds = (all_inc_probs >= 0.5).astype(np.float64)
        inc_f1 = f1_score(all_inc_labels, inc_preds, zero_division=0)
    else:
        inc_auc = float('nan')
        inc_f1 = float('nan')

    # ---------- 视频级来源召回（Video-level Source Recall） ----------
    video_level_recall = {}
    for i, name in enumerate(label_names):
        if video_support[i] > 0:
            video_level_recall[name] = float(video_tp[i] / video_support[i])
        else:
            video_level_recall[name] = float('nan')

    metrics = {
        # 表1 主任务
        'frame_accuracy': frame_acc,
        'frame_f1_samples': frame_f1,
        'frame_f1_macro': frame_f1_macro,
        'frame_ap': frame_ap,
        'per_class_f1': per_class_f1,
        # 表2 辅任务
        'inconsistency_auc': inc_auc,
        'inconsistency_f1': inc_f1,
        # 视频级来源召回（审稿友好）
        'video_level_source_recall': video_level_recall,
        'video_level_source_recall_mean': float(np.nanmean(list(video_level_recall.values()))) if video_level_recall else float('nan'),
    }
    return {'video_level_metrics': metrics, 'frame_level_predictions': []}

