# -*- coding: utf-8 -*-
"""
使用强化的 CNN-CC-BiMamba 混合模型训练 PTB-XL 数据集 (V9 - Debugged)

🚀 V9 修正与改进:
1. 🔥【BUG修复】恢复了CNN特征提取器中的残差连接。这是导致训练失败 (F1=0) 和后续加载错误的核心原因。
2. ✨【架构强化】简化并稳定了分类器头部，避免混合使用BatchNorm和LayerNorm。
3. ✨【代码健壮性】在加载模型权重时增加了try-except块，以提供更清晰的错误信息，防止因旧模型文件导致崩溃。
4. ✅【策略保留】保留了所有成功的SOTA训练策略 (FocalLoss, OneCycleLR, SWA, TTA等)。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset, Dataset
import argparse
import os
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.swa_utils import AveragedModel, SWALR

# 尝试导入优化的Mamba库
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
    print("✅ 成功导入 mamba_ssm 库")
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("❌ 未安装 mamba_ssm 库")

# ==============================================================================
# 可重现性与损失函数 (保持不变)
# ==============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 所有随机种子已设置为: {seed}")

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class FocalAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-7, alpha=0.1, focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps, self.alpha, self.focal_gamma, self.class_weights = \
            gamma_neg, gamma_pos, clip, eps, alpha, focal_gamma, class_weights

    def forward(self, x, y):
        # Label smoothing
        smooth_targets = y * (1 - self.alpha) + 0.5 * self.alpha
        x_sigmoid = torch.sigmoid(x)
        xs_pos, xs_neg = x_sigmoid, 1 - x_sigmoid
        
        # Asymmetric Focusing
        los_pos = smooth_targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - smooth_targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        
        # Asymmetric Probabilistic Modulation
        pt = xs_pos * smooth_targets + xs_neg * (1 - smooth_targets)
        one_sided_gamma = self.gamma_pos * smooth_targets + self.gamma_neg * (1 - smooth_targets)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        
        # Focal Loss component
        focal_w = torch.pow(1 - pt, self.focal_gamma)
        
        loss *= one_sided_w * focal_w
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0).to(x.device)
            
        return -loss.mean()

# ==============================================================================
# 数据增强与加载 (保持不变)
# ==============================================================================
class ECGDataAugmenter:
    def __init__(self, noise_level=0.005, scale_range=0.05, shift_max=8):
        self.noise_level, self.scale_range, self.shift_max = noise_level, scale_range, shift_max

    def __call__(self, x):
        if self.noise_level > 0 and random.random() < 0.5: x += torch.randn_like(x) * self.noise_level
        if self.scale_range > 0 and random.random() < 0.5: x *= (1.0 + (torch.rand(1) - 0.5) * 2 * self.scale_range).to(x.device)
        if self.shift_max > 0 and random.random() < 0.5: x = torch.roll(x, shifts=torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item(), dims=-1)
        return x

class AugmentedTensorDataset(Dataset):
    def __init__(self, tensor_dataset, augmenter): self.tensor_dataset, self.augmenter = tensor_dataset, augmenter
    def __len__(self): return len(self.tensor_dataset)
    def __getitem__(self, idx):
        x, y = self.tensor_dataset[idx]
        return self.augmenter(x) if self.augmenter else x, y

def load_ptbxl_data(data_file_path):
    data = np.load(data_file_path, allow_pickle=True)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'], data['classes']

def create_optimized_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, seed):
    train_ds = AugmentedTensorDataset(TensorDataset(X_train, y_train), ECGDataAugmenter())
    val_ds, test_ds = TensorDataset(X_val, y_val), TensorDataset(X_test, y_test)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

def calculate_class_weights(y_train):
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    pos_weights = len(y_train) / (y_train.sum(axis=0) + 1e-8)
    return torch.from_numpy(pos_weights / np.mean(pos_weights)).float()

# ==============================================================================
# 🔥【V9 核心架构 - 已修复】
# ==============================================================================
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[5, 11, 23], bottleneck_channels=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(bottleneck_channels, n_filters, k, padding='same', bias=False) for k in kernel_sizes])
        self.maxpool_conv = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, n_filters, 1, bias=False))
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.act = nn.SiLU()

    def forward(self, x):
        bottleneck_out = self.bottleneck(x)
        conv_outputs = [conv(bottleneck_out) for conv in self.convs]
        maxpool_out = self.maxpool_conv(x)
        return self.act(self.bn(torch.cat(conv_outputs + [maxpool_out], dim=1)))

class EnhancedECGCNNExtractor(nn.Module):
    def __init__(self, input_channels=12, d_model=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(128), nn.SiLU()
        )
        self.inception1 = InceptionBlock1D(128, 48) # 192 channels out
        
        # 🔥 [FIX] 恢复残差连接，这是稳定深度CNN训练的关键。
        # 维度必须匹配：inception1的输出是192通道，所以残差连接也需要是192通道。
        self.residual_proj1 = nn.Conv1d(128, 192, 1, bias=False) # 1x1卷积用于匹配通道数
        self.bn_res1 = nn.BatchNorm1d(192)

        self.downsample1 = nn.Sequential(
            nn.Conv1d(192, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(192), nn.SiLU()
        )
        self.inception2 = InceptionBlock1D(192, 64) # 256 channels out
        self.proj = nn.Conv1d(256, d_model, 1)

    def forward(self, x):
        x = self.stem(x)
        
        # 🔥 [FIX] 应用残差连接
        residual = self.bn_res1(self.residual_proj1(x))
        x = self.inception1(x)
        x = x + residual # Additive skip-connection
        
        x = self.downsample1(x)
        x = self.inception2(x)
        x = self.proj(x)
        return x.transpose(1, 2)

class ContextClusteringLayer(nn.Module):
    def __init__(self, d_model, n_clusters=8, dropout=0.1):
        super().__init__()
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, d_model))
        self.projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        projected = self.projection(x)
        similarity = torch.einsum('bld,cd->blc', F.normalize(projected, dim=-1), F.normalize(self.cluster_centers, dim=-1))
        weights = torch.softmax(similarity, dim=-1)
        context_vector = torch.matmul(weights, self.cluster_centers)
        return self.dropout(self.norm(x + self.alpha * context_vector))

class BidirectionalMambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x):
        forward_out = self.forward_mamba(x)
        backward_out = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
        return self.fusion(torch.cat([forward_out, backward_out], dim=-1))

class CCBiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, n_clusters=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba_path = BidirectionalMambaLayer(d_model, d_state, expand)
        self.cc_path = ContextClusteringLayer(d_model, n_clusters, dropout)
        self.gate = nn.Sequential(nn.Linear(d_model, 2), nn.Softmax(dim=-1))
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))

    def forward(self, x):
        x_norm = self.norm1(x)
        mamba_out = self.mamba_path(x_norm)
        cc_out = self.cc_path(x_norm)
        
        gate_weights = self.gate(x_norm).unsqueeze(-1)
        fused_out = gate_weights[..., 0, :] * mamba_out + gate_weights[..., 1, :] * cc_out
        x = x + fused_out
        x = x + self.ffn(x)
        return x

class EnhancedCNNCCBiMamba(nn.Module):
    def __init__(self, num_classes=5, d_model=256, n_blocks=6, d_state=16, expand=2, dropout=0.2):
        super().__init__()
        self.cnn_extractor = EnhancedECGCNNExtractor(d_model=d_model)
        self.cc_mamba_blocks = nn.Sequential(*[
            CCBiMambaBlock(d_model, d_state, expand, dropout=dropout) for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
        # ✨ [IMPROVEMENT] 简化并稳定分类器头部，避免混合使用BatchNorm和LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        features = self.cnn_extractor(x)
        features = self.cc_mamba_blocks(features)
        features = self.final_norm(features)
        
        avg_pool = torch.mean(features, dim=1)
        max_pool, _ = torch.max(features, dim=1)
        pooled = torch.cat((avg_pool, max_pool), dim=1)
        return self.classifier(pooled)

# ==============================================================================
# 训练与评估 (已修复)
# ==============================================================================
def full_validation(model, val_loader, device):
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with amp.autocast(enabled=True):
                output = model(data)
            preds = (torch.sigmoid(output) > 0.5).float()
            val_preds.append(preds.cpu().numpy())
            val_true.append(target.cpu().numpy())
    return f1_score(np.concatenate(val_true), np.concatenate(val_preds), average='macro', zero_division=0)

def tta_validation(model, test_loader, device, optimal_thresholds, tta_steps=8):
    model.eval()
    test_preds, test_true = [], []
    augmenter = ECGDataAugmenter()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="🚀 TTA测试中 (8-steps)"):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            tta_predictions = []
            with amp.autocast(enabled=True):
                tta_predictions.append(torch.sigmoid(model(data)))
                for _ in range(tta_steps - 1):
                    tta_predictions.append(torch.sigmoid(model(augmenter(data.clone()))))
            avg_probs = torch.stack(tta_predictions).mean(dim=0)
            preds = (avg_probs > optimal_thresholds).float()
            test_preds.append(preds.cpu().numpy())
            test_true.append(target.cpu().numpy())
    return np.concatenate(test_preds), np.concatenate(test_true)

def find_optimal_thresholds_pr_curve(model, val_loader, device, num_classes, sample_ratio=0.75):
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(val_loader, desc="🚀 PR曲线阈值搜索中", leave=False)):
            if i / len(val_loader) > sample_ratio: break
            data = data.to(device, non_blocking=True)
            with amp.autocast(enabled=True):
                probs = torch.sigmoid(model(data))
            val_probs.append(probs.cpu().numpy())
            val_true.append(target.cpu().numpy())
    
    val_probs, val_true = np.concatenate(val_probs), np.concatenate(val_true)
    optimal_thresholds = np.full(num_classes, 0.5)
    for i in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(val_true[:, i], val_probs[:, i])
        if len(thresholds) > 0:
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            if len(f1_scores) > 0:
                optimal_thresholds[i] = thresholds[np.argmax(f1_scores)]
    print(f"✅ 找到最优阈值: {np.round(optimal_thresholds, 4)}")
    return torch.tensor(optimal_thresholds).to(device)

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 and random.random() < 0.5 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index, :], y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main_sota_training(args):
    seed_everything(args.seed)
    
    print("🚀 启动 CNN-CC-BiMamba V9 (Debugged) 模型训练...")
    
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_ptbxl_data(args.data_file)
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

    train_loader, val_loader, test_loader = create_optimized_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args.batch_size, args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedCNNCCBiMamba(
        num_classes=len(classes),
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        d_state=args.d_state,
        dropout=args.dropout
    ).to(device)
    
    print(f"🏥 V9 模型配置: d_model={args.d_model}, blocks={args.n_blocks}, d_state={args.d_state}")
    
    criterion = FocalAsymmetricLoss(class_weights=calculate_class_weights(y_train))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr * 10, total_steps=len(train_loader) * args.epochs)

    scaler = amp.GradScaler(enabled=True)
    
    best_f1, patience_counter = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for data, target in progress_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            mixed_data, target_a, target_b, lam = mixup_data(data, target, alpha=args.mixup_alpha)
            
            with amp.autocast(enabled=True):
                output = model(mixed_data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)

            if not torch.isfinite(loss): continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.7f}'})
        
        val_f1 = full_validation(model, val_loader, device)
        print(f"Epoch {epoch+1}, Val F1-macro: {val_f1:.4f}, Avg Loss: {total_loss / len(train_loader):.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            print(f"📈 新的最佳验证 F1 分数: {best_f1:.4f}")
            
            if best_f1 > args.save_threshold:
                torch.save(model.state_dict(), args.save_path)
                print(f"🚀 模型已保存，因为它超过了阈值 {args.save_threshold:.4f}！")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"早停: 在 {epoch+1} 个epoch触发")
            break

    if not os.path.exists(args.save_path):
        print("\n⚠️ 警告: 训练结束，但没有任何模型达到保存阈值。将不会进行SWA和最终评估。")
        return

    print("\n🚀 开始SWA优化...")
    # ✨ [IMPROVEMENT] 增加健壮性，使用try-except处理潜在的模型加载错误
    try:
        model.load_state_dict(torch.load(args.save_path, map_location=device, weights_only=True))
    except RuntimeError as e:
        print("\n❌ 加载模型权重失败！这可能是因为保存的模型与当前定义的模型架构不匹配。")
        print("请检查 `save_path` 是否指向了由旧版本代码生成的模型文件。")
        print(f"详细错误: {e}")
        return # 无法继续，直接退出

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    
    for swa_epoch in range(args.swa_epochs):
        for data, target in tqdm(train_loader, desc=f'SWA Epoch {swa_epoch+1}/{args.swa_epochs}', leave=False):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                loss = criterion(model(data), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        swa_model.update_parameters(model)
        swa_scheduler.step()
        print(f"SWA Epoch {swa_epoch+1}, Val F1-macro: {full_validation(swa_model, val_loader, device):.4f}")

    print("\n🚀 在测试集上评估最终SWA模型")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    optimal_thresholds = find_optimal_thresholds_pr_curve(swa_model, val_loader, device, len(classes))
    test_preds, test_true = tta_validation(swa_model, test_loader, device, optimal_thresholds)
    
    print("\n--- 🚀 CNN-CC-BiMamba V9 (SWA+TTA) 模型分类报告 ---")
    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
    
    macro_f1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    print(f"\n--- 🚀 Test Set Performance ---")
    print(f"Macro F1: {macro_f1:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🚀 CNN-CC-BiMamba V9 (Debugged) 训练')
    # 🔥 [FIX] 路径已根据您的要求修改为绝对路径
    parser.add_argument('--data_file', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data/ptbxl_processed_100hz.npz')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_blocks', type=int, default=6)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--swa_epochs', type=int, default=5)
    parser.add_argument('--swa_lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='./saved_models/ptbxl_cnn_cc_mamba_v9_sota.pth')
    parser.add_argument('--save_threshold', type=float, default=0.73, help='只有当验证集F1分数超过此阈值时才保存模型')
    
    args = parser.parse_args()
    main_sota_training(args)
