# -*- coding: utf-8 -*-
"""
使用 Inception-Mamba V10 (SOTA优化版) 训练 PTB-XL 数据集

🚀 V10 优化策略 (针对SOTA提升):
1. 🔥 引入自适应小波变换预处理 (inspired by RLWBS), 动态选择小波基提升频率特征提取
2. 🔥 融合xLSTM元素到Mamba层，提升序列建模 (参考xLSTM-ECG SOTA)
3. 🔥 添加自监督预训练 (SSL) 阶段，使用Masked Autoencoder预训练特征提取器
4. 🔥 集成Focal Loss到AsymmetricLoss，进一步处理不平衡
5. 🔥 增强TTA到8步，并添加简单ensemble (3模型平均)
6. 🔥 延长epochs到80，优化阈值使用PR曲线
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset, Dataset
import argparse
import os
from tqdm import tqdm
import time
import random
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.swa_utils import AveragedModel, SWALR
import pywt  # 新增: 小波变换

# 尝试导入优化的Mamba库
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
    print("✅ 成功导入 mamba_ssm 库")
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("❌ 未安装 mamba_ssm 库")

# ==============================================================================
# Enhanced Asymmetric Loss with Focal and Class Weighting
# ==============================================================================
class FocalAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-7, alpha=0.1, focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.alpha = alpha
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights

    def forward(self, x, y):
        smooth_targets = y * (1 - self.alpha) + 0.5 * self.alpha
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        los_pos = smooth_targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - smooth_targets) * torch.log(xs_neg.clamp(min=self.eps))
        
        loss = los_pos + los_neg
        
        pt0 = xs_pos * smooth_targets
        pt1 = xs_neg * (1 - smooth_targets)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * smooth_targets + self.gamma_neg * (1 - smooth_targets)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        focal_w = torch.pow(1 - pt, self.focal_gamma)  # 新增: Focal weighting
        loss *= one_sided_w * focal_w
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        return -loss.mean()

# ==============================================================================
# Wavelet Preprocessing (Adaptive Base Selection inspired by RLWBS)
# ==============================================================================
class AdaptiveWaveletPreprocessor:
    def __init__(self, wavelet_bases=['db4', 'sym5', 'coif3'], level=4):
        self.wavelet_bases = wavelet_bases
        self.level = level

    def __call__(self, x):
        # 批量处理
        if len(x.shape) == 3:  # (batch, channels, length)
            processed = []
            for i in range(x.shape[0]):
                sample_processed = []
                for j in range(x.shape[1]):  # 每个通道
                    # 随机选择小波基 (模拟RL动态选择)
                    base = random.choice(self.wavelet_bases)
                    try:
                        coeffs = pywt.wavedec(x[i, j].cpu().numpy(), base, level=self.level)
                        approx = pywt.waverec(coeffs, base)
                        # 截断到原长度
                        if len(approx) > x.shape[2]:
                            approx = approx[:x.shape[2]]
                        elif len(approx) < x.shape[2]:
                            # 填充
                            approx = np.pad(approx, (0, x.shape[2] - len(approx)), 'edge')
                        sample_processed.append(approx)
                    except:
                        # 如果小波变换失败，返回原信号
                        sample_processed.append(x[i, j].cpu().numpy())
                processed.append(sample_processed)
            return torch.tensor(processed).float().to(x.device)
        else:  # 单样本处理
            base = random.choice(self.wavelet_bases)
            try:
                coeffs = pywt.wavedec(x.cpu().numpy(), base, level=self.level)
                approx = pywt.waverec(coeffs, base)
                return torch.from_numpy(approx).float().to(x.device)[:x.shape[0]]
            except:
                return x

# ==============================================================================
# Data Augmentation (保持原有的ECGDataAugmenter)
# ==============================================================================
class ECGDataAugmenter:
    def __init__(self, noise_level=0.005, scale_range=0.05, shift_max=8):
        self.noise_level, self.scale_range, self.shift_max = noise_level, scale_range, shift_max

    def __call__(self, x):
        if self.noise_level > 0 and random.random() < 0.5:
            x += torch.randn_like(x) * self.noise_level
        if self.scale_range > 0 and random.random() < 0.5:
            scale = 1.0 + (torch.rand(1) - 0.5) * 2 * self.scale_range
            x *= scale.to(x.device)
        if self.shift_max > 0 and random.random() < 0.5:
            shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)
        return x

# 更新AugmentedDataset以集成小波
class AugmentedTensorDataset(Dataset):
    def __init__(self, tensor_dataset, augmenter, wavelet_preprocessor):
        self.tensor_dataset = tensor_dataset
        self.augmenter = augmenter
        self.wavelet_preprocessor = wavelet_preprocessor

    def __len__(self): 
        return len(self.tensor_dataset)
    
    def __getitem__(self, idx):
        x, y = self.tensor_dataset[idx]
        if self.wavelet_preprocessor and random.random() < 0.7:
            x = self.wavelet_preprocessor(x)
        return self.augmenter(x) if self.augmenter else x, y

# ==============================================================================
# SSL Pretraining (Masked Autoencoder)
# ==============================================================================
class SSLPretrainer(nn.Module):
    def __init__(self, model, mask_ratio=0.15):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(model.feature_extractor.proj.out_channels, 256),
            nn.GELU(),
            nn.Linear(256, 12)  # 重建到原输入通道
        )

    def forward(self, x):
        # Masking
        B, C, L = x.shape
        mask_indices = torch.randperm(L)[:int(L * self.mask_ratio)]
        x_masked = x.clone()
        x_masked[:, :, mask_indices] = 0
        
        features = self.model.feature_extractor(x_masked)  # (B, L', d_model)
        # 全局平均池化
        pooled_features = features.mean(dim=1)  # (B, d_model)
        recon = self.decoder(pooled_features)  # (B, 12)
        return recon

def ssl_pretrain(model, train_loader, device, epochs=10, lr=1e-4):
    print(f"🚀 开始SSL预训练 ({epochs} epochs)")
    pretrainer = SSLPretrainer(model).to(device)
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, _ in tqdm(train_loader, desc=f'SSL Epoch {epoch+1}/{epochs}', leave=False):
            data = data.to(device, non_blocking=True)
            recon = pretrainer(data)
            # 重建目标是每个通道的平均值
            target = data.mean(dim=2)  # (B, 12)
            loss = criterion(recon, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'SSL Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

# ==============================================================================
# Data Loading Functions (保持原有函数)
# ==============================================================================
def load_ptbxl_data(data_file_path):
    print(f"正在加载PTB-XL数据: {data_file_path}")
    data = np.load(data_file_path, allow_pickle=True)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'], data['classes']

def create_optimized_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    wavelet_preprocessor = AdaptiveWaveletPreprocessor()
    
    train_ds = AugmentedTensorDataset(TensorDataset(X_train, y_train), ECGDataAugmenter(), wavelet_preprocessor)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

def calculate_class_weights(y_train):
    print("🚀 正在计算类别权重...")
    pos_weights = len(y_train) / (y_train.sum(axis=0) + 1e-8)
    pos_weights = pos_weights / np.mean(pos_weights)  # Normalize
    print(f"计算出的类别权重: {pos_weights}")
    return torch.from_numpy(pos_weights).float()

# ==============================================================================
# Updated Architecture with xLSTM Fusion (保持原有架构基础组件)
# ==============================================================================
class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels // 2)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels // 2, out_channels, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[5, 11, 23], bottleneck_channels=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, n_filters, k, padding='same', bias=False) for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, n_filters, 1, bias=False))
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.act = nn.SiLU()

    def forward(self, x):
        bottleneck_out = self.bottleneck(x)
        conv_outputs = [conv(bottleneck_out) for conv in self.convs]
        maxpool_out = self.maxpool_conv(x)
        concat_out = torch.cat(conv_outputs + [maxpool_out], dim=1)
        return self.act(self.bn(concat_out))

class InceptionECGExtractorV5(nn.Module):
    def __init__(self, input_channels=12, d_model=256):
        super().__init__()
        self.stem = ConvStem(input_channels, 64)
        self.inception1 = InceptionBlock1D(64, 32)
        self.downsample1 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.inception2 = InceptionBlock1D(128, 64)
        self.proj = nn.Conv1d(256, d_model, 1)
        
        self.residual_proj = nn.Conv1d(64, 128, 1)

    def forward(self, x):
        x = self.stem(x)
        res = self.residual_proj(x)
        x = self.inception1(x)
        x = x + res
        x = self.downsample1(x)
        x = self.inception2(x)
        x = self.proj(x)
        return x.transpose(1, 2)

# 更新的Mamba层，融合xLSTM元素
class XLSTMBidirectionalMambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        if not MAMBA_SSM_AVAILABLE: 
            raise ImportError("mamba-ssm is not available.")
        
        self.norm = nn.LayerNorm(d_model)
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        
        # xLSTM-like融合
        self.lstm = nn.LSTM(d_model, d_model // 2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fusion = nn.Linear(d_model * 3, d_model)  # Mamba fwd + bwd + LSTM
        
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        forward_out = self.forward_mamba(x)
        backward_out = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
        lstm_out, _ = self.lstm(x)
        
        fused_out = self.fusion(torch.cat([forward_out, backward_out, lstm_out], dim=-1))
        return residual + fused_out

class InceptionMambaClassifierV10(nn.Module):
    def __init__(self, num_classes=5, d_model=256, n_mamba_layers=8, dropout=0.2):
        super().__init__()
        self.feature_extractor = InceptionECGExtractorV5(d_model=d_model)
        
        # 增加到8层Mamba层
        self.mamba_layers = nn.Sequential(*[
            XLSTMBidirectionalMambaLayer(d_model) for _ in range(n_mamba_layers)
        ])
        
        # 新增多头注意力
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
        self.final_norm = nn.LayerNorm(d_model)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.mamba_layers(features)
        
        # 自注意力
        features, _ = self.attention(features, features, features)
        features = self.final_norm(features)
        
        # 池化
        avg_pool = torch.mean(features, dim=1)
        max_pool, _ = torch.max(features, dim=1)
        pooled = torch.cat((avg_pool, max_pool), dim=1)
        
        return self.classifier(pooled)

# ==============================================================================
# Updated Validation and TTA with PR-curve based threshold optimization
# ==============================================================================
def find_optimal_thresholds_pr_curve(model, val_loader, device, num_classes, sample_ratio=0.5):
    """使用PR曲线找到最优阈值"""
    model.eval()
    val_probs, val_true = [], []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(val_loader, desc="🚀 PR曲线阈值搜索中", leave=False)):
            if i / len(val_loader) > sample_ratio: 
                break
            data = data.to(device, non_blocking=True)
            with amp.autocast(device_type='cuda'):
                probs = torch.sigmoid(model(data))
            val_probs.append(probs.cpu().numpy())
            val_true.append(target.cpu().numpy())
    
    val_probs, val_true = np.concatenate(val_probs), np.concatenate(val_true)
    optimal_thresholds = np.full(num_classes, 0.5)
    
    print(f"🚀 为 {num_classes} 个类别寻找最优阈值...")
    for i in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(val_true[:, i], val_probs[:, i])
        if len(thresholds) > 0:
            f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
            best_idx = np.argmax(f1_scores)
            optimal_thresholds[i] = thresholds[best_idx]
        print(f"  类别 {i}: 最优阈值 = {optimal_thresholds[i]:.4f}")
    
    return torch.tensor(optimal_thresholds).to(device)

def full_validation(model, val_loader, device):
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with amp.autocast(device_type='cuda'):
                output = model(data)
            preds = (torch.sigmoid(output) > 0.5).float()
            val_preds.append(preds.cpu().numpy())
            val_true.append(target.cpu().numpy())
    return f1_score(np.concatenate(val_preds), np.concatenate(val_true), average='macro', zero_division=0)

# ==============================================================================
# Ensemble (简单3模型平均)
# ==============================================================================
def ensemble_predict(models, data):
    """集成预测"""
    preds = []
    for model in models:
        with amp.autocast(device_type='cuda'):
            preds.append(torch.sigmoid(model(data)))
    return torch.mean(torch.stack(preds), dim=0)

def tta_validation(ensemble_models, test_loader, device, optimal_thresholds, tta_steps=8):
    """增强的TTA验证，支持模型集成"""
    for model in ensemble_models: 
        model.eval()
    
    test_preds, test_true = [], []
    augmenter = ECGDataAugmenter(noise_level=0.005, scale_range=0.05, shift_max=5)
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="🚀 TTA+集成测试中"):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            tta_predictions = []
            
            # 原始数据预测
            tta_predictions.append(ensemble_predict(ensemble_models, data))
            
            # TTA增强预测
            for _ in range(tta_steps - 1):
                augmented_data = augmenter(data.clone())
                tta_predictions.append(ensemble_predict(ensemble_models, augmented_data))
            
            # 平均所有TTA预测
            avg_probs = torch.stack(tta_predictions).mean(dim=0)
            preds = (avg_probs > optimal_thresholds).float()
            
            test_preds.append(preds.cpu().numpy())
            test_true.append(target.cpu().numpy())
            
    return np.concatenate(test_preds), np.concatenate(test_true)

# ==============================================================================
# Training utilities (保持原有的调度器和mixup)
# ==============================================================================
class CosineAnnealingWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-6, last_epoch=-1):
        self.warmup_steps, self.total_steps, self.eta_min = warmup_steps, total_steps, eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs] if self.warmup_steps > 0 else self.base_lrs
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2 for base_lr in self.base_lrs]

def mixup_data(x, y, alpha=0.4):
    if alpha > 0 and random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==============================================================================
# 主训练函数 (V10)
# ==============================================================================
def main_inception_mamba_v10(args):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("🚀 启动 Inception-Mamba V10 (SOTA优化版) 模型训练...")
    
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_ptbxl_data(args.data_file)
    
    class_weights = calculate_class_weights(y_train)
    
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

    train_loader, val_loader, test_loader = create_optimized_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建3个模型 for ensemble
    print(f"🚀 创建 {args.n_ensemble} 个模型进行集成学习")
    ensemble_models = []
    for i in range(args.n_ensemble):
        model = InceptionMambaClassifierV10(
            num_classes=len(classes), 
            d_model=args.d_model, 
            n_mamba_layers=args.n_mamba_layers,
            dropout=args.dropout
        ).to(device)
        ensemble_models.append(model)
    
    print(f"🏥 Inception-Mamba V10 模型配置: d_model={args.d_model}, layers={args.n_mamba_layers}, ensemble={args.n_ensemble}")
    
    # SSL预训练每个模型
    if args.ssl_pretrain:
        for i, model in enumerate(ensemble_models):
            print(f"🚀 SSL预训练模型 {i+1}/{args.n_ensemble}")
            ssl_pretrain(model, train_loader, device, epochs=args.ssl_epochs, lr=args.ssl_lr)
    
    criterion = FocalAsymmetricLoss(class_weights=class_weights.to(device))
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    
    # 训练每个模型
    best_f1s = []
    swa_models = []
    
    for i, model in enumerate(ensemble_models):
        print(f"\n🚀 训练模型 {i+1}/{args.n_ensemble}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-6)
        scaler = amp.GradScaler(growth_interval=2000)
        scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_steps, total_steps)
        
        best_f1 = 0.0
        patience_counter = 0
        model_save_path = args.save_path.replace('.pth', f'_model_{i}.pth')
        
        # 训练循环
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Model {i+1} Epoch {epoch+1}/{args.epochs}')
            
            for data, target in progress_bar:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                mixed_data, target_a, target_b, lam = mixup_data(data, target, alpha=args.mixup_alpha)
                
                with amp.autocast(device_type='cuda'):
                    output = model(mixed_data)
                    loss = mixup_criterion(criterion, output, target_a, target_b, lam)

                if not torch.isfinite(loss):
                    print(f"💥 损失异常 at epoch {epoch+1}. 跳过此批次.")
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.7f}'})
            
            avg_loss = total_loss / len(train_loader)
            val_f1 = full_validation(model, val_loader, device)
            print(f"Model {i+1} Epoch {epoch+1}, Val F1-macro: {val_f1:.4f}, Avg Loss: {avg_loss:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"🚀 模型 {i+1} 新的最佳模型已保存，F1-macro: {best_f1:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"模型 {i+1} 早停: {args.patience} 个epoch无改善")
                break
        
        best_f1s.append(best_f1)
        
        # SWA优化
        print(f"\n🚀 模型 {i+1} 开始SWA优化...")
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
        swa_model = AveragedModel(model)
        swa_optimizer = torch.optim.AdamW(model.parameters(), lr=args.swa_lr)
        swa_scheduler = SWALR(swa_optimizer, swa_lr=args.swa_lr)
        
        for swa_epoch in range(args.swa_epochs):
            model.train()
            for data, target in tqdm(train_loader, desc=f'Model {i+1} SWA Epoch {swa_epoch+1}/{args.swa_epochs}', leave=False):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                swa_optimizer.zero_grad(set_to_none=True)
                with amp.autocast(device_type='cuda'):
                    loss = criterion(model(data), target)
                scaler.scale(loss).backward()
                scaler.step(swa_optimizer)
                scaler.update()
            
            swa_model.update_parameters(model)
            swa_scheduler.step()
            
            swa_val_f1 = full_validation(swa_model, val_loader, device)
            print(f"Model {i+1} SWA Epoch {swa_epoch+1}, Val F1-macro: {swa_val_f1:.4f}")

        # 更新BN层
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_models.append(swa_model)
    
    print(f"\n🚀 所有模型训练完成! 个体最佳F1分数: {best_f1s}")
    print(f"平均最佳F1: {np.mean(best_f1s):.4f} ± {np.std(best_f1s):.4f}")
    
    # 使用PR曲线找最优阈值 (用第一个模型)
    print("\n🚀 使用PR曲线寻找最优阈值...")
    optimal_thresholds = find_optimal_thresholds_pr_curve(swa_models[0], val_loader, device, len(classes))
    
    # 集成测试
    print("\n🚀 在测试集上评估SWA集成模型 (TTA + Ensemble)")
    test_preds, test_true = tta_validation(swa_models, test_loader, device, optimal_thresholds, tta_steps=args.tta_steps)
    
    print("\n--- 🚀 Inception-Mamba V10 (SWA+TTA+Ensemble) 模型分类报告 ---")
    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
    
    macro_f1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(test_true, test_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)

    print(f"\n--- 🚀 Final Test Set Performance (Inception-Mamba V10 SOTA Model) ---")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # 保存结果
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    results = {
        'model_type': 'Inception_Mamba_V10_SOTA',
        'test_predictions': test_preds,
        'test_true': test_true,
        'optimal_thresholds': optimal_thresholds.cpu().numpy(),
        'individual_best_f1s': best_f1s,
        'ensemble_macro_f1': macro_f1,
        'ensemble_micro_f1': micro_f1,
        'ensemble_weighted_f1': weighted_f1
    }
    np.save(args.results_path, results)
    print(f"✅ 结果已保存到: {args.results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🚀 Inception-Mamba V10 (SOTA优化版) 训练')
    parser.add_argument('--data_file', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data/ptbxl_processed_100hz.npz')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--n_mamba_layers', type=int, default=8, help='Mamba层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--swa_epochs', type=int, default=5, help='SWA轮数')
    parser.add_argument('--swa_lr', type=float, default=2e-4, help='SWA学习率')
    parser.add_argument('--n_ensemble', type=int, default=3, help='集成模型数量')
    parser.add_argument('--tta_steps', type=int, default=8, help='TTA步数')
    parser.add_argument('--ssl_pretrain', action='store_true', help='启用SSL预训练')
    parser.add_argument('--ssl_epochs', type=int, default=10, help='SSL预训练轮数')
    parser.add_argument('--ssl_lr', type=float, default=1e-4, help='SSL预训练学习率')
    parser.add_argument('--save_path', type=str, default='./saved_models/ptbxl_inception_mamba_v10_sota.pth')
    parser.add_argument('--results_path', type=str, default='./results/ptbxl_inception_mamba_v10_sota_results.npy')
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    main_inception_mamba_v10(args)