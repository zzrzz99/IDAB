"""
训练器核心逻辑
拆分自 utils/trainer.py，避免循环导入问题
"""
import numpy as np
import torch
import torch.distributed as dist


def _smooth_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return labels
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def _apply_mixup(images, labels, alpha):
    if alpha <= 0.0 or images.size(0) < 2:
        return images, labels
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    mixed_labels = lam * labels + (1.0 - lam) * labels[index]
    return mixed_images, mixed_labels


def _rand_bbox(size, lam):
    """生成CutMix裁剪框 (batch, T, C, H, W)"""
    W = size[-1]
    H = size[-2]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def _apply_cutmix(images, labels, alpha):
    if alpha <= 0.0 or images.size(0) < 2:
        return images, labels
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    x1, y1, x2, y2 = _rand_bbox(images.shape, lam)
    if x1 == x2 or y1 == y2:
        return images, labels

    images[:, :, :, y1:y2, x1:x2] = images[index, :, :, y1:y2, x1:x2]
    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    mixed_labels = lam_adj * labels + (1.0 - lam_adj) * labels[index]
    return images, mixed_labels


def train_epoch(model, train_loader, optimizer, criterion, device, epoch=None, scaler=None,
                use_amp=False, label_smoothing: float = 0.0,
                mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0,
                distributed: bool = False, use_inconsistency_labels: bool = False,
                dual_task: bool = False, lambda_inc: float = 0.0, lambda_transition: float = 0.5,
                margin_transition: float = 1.0, use_inc: bool = True):
    model.train()
    loss_sum = 0.0
    sample_count = 0.0

    for batch in train_loader:
        if isinstance(batch, dict):
            images = batch['frames'].to(device, non_blocking=True)
            frame_labels = batch['frame_labels'].to(device, non_blocking=True)
            inconsistency_labels = batch['inconsistency_labels'].to(device, non_blocking=True)
            labels = frame_labels  # 占位，compute_loss 用 frame_labels
        else:
            if len(batch) == 4:
                images, labels, frame_labels, _ = batch
            else:
                images, labels, _ = batch
                frame_labels = None
            inconsistency_labels = None
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if frame_labels is not None:
                frame_labels = frame_labels.to(device, non_blocking=True)

        if not dual_task:
            # 不一致性任务不做 mixup/cutmix，保持 0/1 语义
            if not use_inconsistency_labels:
                if cutmix_alpha > 0.0 and mixup_alpha > 0.0:
                    if np.random.rand() < 0.5:
                        images, labels = _apply_cutmix(images, labels, cutmix_alpha)
                    else:
                        images, labels = _apply_mixup(images, labels, mixup_alpha)
                elif cutmix_alpha > 0.0:
                    images, labels = _apply_cutmix(images, labels, cutmix_alpha)
                elif mixup_alpha > 0.0:
                    images, labels = _apply_mixup(images, labels, mixup_alpha)
            labels = _smooth_labels(labels, label_smoothing)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            compute_loss_fn = getattr(model, 'compute_loss', None)
            if compute_loss_fn is None and hasattr(model, 'module'):
                compute_loss_fn = getattr(model.module, 'compute_loss', None)

            if compute_loss_fn is not None:
                if dual_task:
                    loss, loss_dict = compute_loss_fn(
                        images, labels, epoch,
                        frame_labels=frame_labels,
                        inconsistency_labels=inconsistency_labels,
                        lambda_inc=lambda_inc,
                        lambda_transition=lambda_transition,
                        margin_transition=margin_transition,
                        use_inc=use_inc,
                    )
                else:
                    logits = model(images, apply_sigmoid=False)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    loss, loss_dict = compute_loss_fn(
                        images, labels, epoch, logits=logits,
                        frame_labels=frame_labels if use_inconsistency_labels else None,
                    )
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_dict = {'total_loss': loss.item()}

        batch_size = images.size(0)
        if not torch.isfinite(loss).all():
            optimizer.zero_grad(set_to_none=True)
            continue
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        loss_sum += float(loss_dict['total_loss']) * batch_size
        sample_count += batch_size

    loss_tensor = torch.tensor([loss_sum, sample_count], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_samples = loss_tensor[1].item()
    if global_samples <= 0:
        return float('nan')
    return (loss_tensor[0] / global_samples).item()


def validate(model, val_loader, criterion, device, epoch=None, use_amp=False,
             label_smoothing: float = 0.0, distributed: bool = False,
             use_inconsistency_labels: bool = False, dual_task: bool = False,
             lambda_inc: float = 0.0, lambda_transition: float = 0.5,
             margin_transition: float = 1.0, use_inc: bool = True):
    model.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0
    sample_count = 0.0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                images = batch['frames'].to(device, non_blocking=True)
                frame_labels = batch['frame_labels'].to(device, non_blocking=True)
                inconsistency_labels = batch['inconsistency_labels'].to(device, non_blocking=True)
                labels = frame_labels
            else:
                if len(batch) == 4:
                    images, labels, _, _ = batch
                else:
                    images, labels, _ = batch
                frame_labels = inconsistency_labels = None
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                smooth_labels = _smooth_labels(labels, label_smoothing)

            batch_size = images.size(0)

            with torch.cuda.amp.autocast(enabled=use_amp):
                compute_loss_fn = getattr(model, 'compute_loss', None)
                if compute_loss_fn is None and hasattr(model, 'module'):
                    compute_loss_fn = getattr(model.module, 'compute_loss', None)

                if compute_loss_fn is not None and dual_task:
                    loss, loss_dict = compute_loss_fn(
                        images, labels, epoch,
                        frame_labels=frame_labels,
                        inconsistency_labels=inconsistency_labels,
                        lambda_inc=lambda_inc,
                        lambda_transition=lambda_transition,
                        margin_transition=margin_transition,
                        use_inc=use_inc,
                    )
                    if not torch.isfinite(loss).all():
                        continue
                    out = model(images, apply_sigmoid=True)
                    _, frame_logits, _ = out[:3]
                    preds = (frame_logits > 0.5).float()
                    correct += (preds == frame_labels).sum().item()
                    total += frame_labels.numel()
                elif compute_loss_fn is not None:
                    logits = model(images, apply_sigmoid=False)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    loss, loss_dict = compute_loss_fn(
                        images, smooth_labels, epoch, logits=logits,
                        frame_labels=None,
                    )
                    if not torch.isfinite(loss).all():
                        continue
                    outputs = torch.sigmoid(logits)
                    preds = (outputs > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.numel()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, smooth_labels)
                    loss_dict = {'total_loss': loss.item()}
                    if not torch.isfinite(loss).all():
                        continue
                    preds = (outputs > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.numel()

            if not torch.isfinite(loss).all():
                continue
            val_loss += float(loss_dict['total_loss']) * batch_size
            sample_count += batch_size

    loss_tensor = torch.tensor([val_loss, sample_count], device=device, dtype=torch.float64)
    acc_tensor = torch.tensor([correct, total], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
    global_samples = loss_tensor[1].item()
    if global_samples <= 0:
        return float('nan'), 0.0
    avg_loss = (loss_tensor[0] / global_samples).item()
    global_total = acc_tensor[1].item()
    acc = (acc_tensor[0] / global_total).item() if global_total > 0 else 0.0
    return avg_loss, acc


__all__ = ['train_epoch', 'validate']

