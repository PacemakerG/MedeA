# -*- coding: utf-8 -*-
"""
使用 MeDeA (Multi-disease Decompositional Attention) 模型训练 PTB-XL 数据集

此脚本是一个独立的、可直接运行的文件，包含了从数据加载到模型训练、
评估、保存以及最终可解释性分析的全过程。
"""
import os
import argparse
import random
import time
import copy
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

# ==============================================================================
# 1. 可复现性设置 (Reproducibility)
# ==============================================================================
def seed_everything(seed: int):
    """为所有库设置随机种子以保证结果可复现。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 所有随机种子已设置为: {seed}")

# ==============================================================================
# 2. 数据加载模块 (Data Loading)
# ==============================================================================
class PTBXLDataset(Dataset):
    """为PTB-XL ECG数据创建PyTorch Dataset。"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(data_file_path, batch_size, num_workers):
    """从.npz文件加载数据并创建DataLoaders。"""
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"❌ 数据文件未找到: {data_file_path}")
    
    print(f"⌛️ 正在从 {data_file_path} 加载数据...")
    data = np.load(data_file_path, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    
    train_ds = PTBXLDataset(X_train, y_train)
    val_ds = PTBXLDataset(X_val, y_val)
    test_ds = PTBXLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"✅ 数据加载完成。类别: {classes.tolist()}")
    return train_loader, val_loader, test_loader, classes

def create_dataloaders_cv(data_dir, fold_num, batch_size, num_workers):
    """从十折交叉验证数据文件夹加载指定fold的数据并创建DataLoaders。"""
    # 🔥 修复：确保 data_dir 是目录路径，然后拼接文件名
    data_dir = Path(data_dir)
    
    # 🔥 检查目录是否存在
    if not data_dir.exists():
        raise FileNotFoundError(f"❌ 数据目录未找到: {data_dir}")
    
    if not data_dir.is_dir():
        raise NotADirectoryError(f"❌ 路径不是目录: {data_dir}")
    
    # 🔥 构建完整的文件路径
    data_file_path = data_dir / f"ptbxl_processed_100hz_fold{fold_num}.npz"
    
    # 🔥 检查文件是否存在
    if not data_file_path.exists():
        raise FileNotFoundError(f"❌ 数据文件未找到: {data_file_path}")
    
    print(f"⌛️ 正在从 {data_file_path} 加载第 {fold_num} 折数据...")
    data = np.load(data_file_path, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    
    train_ds = PTBXLDataset(X_train, y_train)
    val_ds = PTBXLDataset(X_val, y_val)
    test_ds = PTBXLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"✅ 第 {fold_num} 折数据加载完成。类别: {classes.tolist()}")
    print(f"   - 训练集: {len(train_ds)} 样本")
    print(f"   - 验证集: {len(val_ds)} 样本") 
    print(f"   - 测试集: {len(test_ds)} 样本")
    
    return train_loader, val_loader, test_loader, classes

# ==============================================================================
# 3. MeDeA 模型定义 (MeDeA Model Definition)
# ==============================================================================
class BasicBlock1D(nn.Module):
    """1D ResNet基础块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class SharedBackbone(nn.Module):
    """共享的1D ResNet骨干网络"""
    def __init__(self, input_channels=12, base_filters=64, dropout=0.1):
        super(SharedBackbone, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_filters, base_filters*2, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, 2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, 2, stride=2, dropout=dropout)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = base_filters * 8
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dropout=0.1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock1D(in_channels, out_channels, stride, downsample, dropout))
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, dropout=dropout))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_maps = self.layer4(x)  # 保留特征图用于注意力
        
        # 全局平均池化
        pooled_features = self.avgpool(feature_maps)
        pooled_features = pooled_features.squeeze(-1)  # (batch_size, feature_dim)
        
        return pooled_features, feature_maps

class AttentionHead(nn.Module):
    """改进的注意力头 - 鼓励分布式注意力"""
    def __init__(self, feature_dim, d_model, dropout=0.1):
        super(AttentionHead, self).__init__()
        self.d_model = d_model
        
        # 注意力机制
        self.query = nn.Linear(feature_dim, d_model)
        self.key = nn.Linear(feature_dim, d_model)
        self.value = nn.Linear(feature_dim, d_model)
        
        # 🔥 添加注意力正则化
        self.attention_regularizer = nn.Parameter(torch.tensor(0.1))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, pooled_features, feature_maps):
        # pooled_features: (batch_size, feature_dim)
        # feature_maps: (batch_size, feature_dim, seq_len)
        
        batch_size, feature_dim, seq_len = feature_maps.shape
        
        # 将特征图重塑为序列格式
        feature_seq = feature_maps.transpose(1, 2)  # (batch_size, seq_len, feature_dim)
        
        # 计算注意力
        Q = self.query(pooled_features).unsqueeze(1)  # (batch_size, 1, d_model)
        K = self.key(feature_seq)  # (batch_size, seq_len, d_model)
        V = self.value(feature_seq)  # (batch_size, seq_len, d_model)
        
        # 注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # 🔥 添加注意力分布正则化
        # 鼓励注意力更均匀分布，避免过度集中
        attention_entropy = -torch.sum(
            F.softmax(attention_scores, dim=-1) * F.log_softmax(attention_scores, dim=-1), 
            dim=-1, keepdim=True
        )
        
        attention_weights = F.softmax(attention_scores, dim=-1)  
        
        # 🔥 可以在损失函数中添加熵正则化
        # loss += -self.attention_regularizer * attention_entropy.mean()
        
        # 应用注意力
        attended_features = torch.matmul(attention_weights, V)  # (batch_size, 1, d_model)
        attended_features = attended_features.squeeze(1)  # (batch_size, d_model)
        
        # 残差连接和层标准化
        attended_features = self.layer_norm(attended_features + self.query(pooled_features))
        attended_features = self.dropout(attended_features)
        
        # 分类
        output = self.classifier(attended_features)
        
        return output, attention_weights.squeeze(1), attention_entropy  # output: (batch_size, 1), weights: (batch_size, seq_len)

class MultiScaleAttentionHead(nn.Module):
    """多尺度注意力头 - 关注不同时间尺度的特征"""
    def __init__(self, feature_dim, d_model, dropout=0.1):
        super(MultiScaleAttentionHead, self).__init__()
        
        # 🔥 多个不同窗口大小的注意力
        self.local_attention = AttentionHead(feature_dim, d_model//2, dropout)   # 局部特征
        self.global_attention = AttentionHead(feature_dim, d_model//2, dropout)  # 全局特征
        
        self.fusion = nn.Linear(d_model, 1)
    
    def forward(self, pooled_features, feature_maps):
        # 局部和全局注意力
        local_out, local_weights, _ = self.local_attention(pooled_features, feature_maps)
        global_out, global_weights, _ = self.global_attention(pooled_features, feature_maps)
        
        # 特征融合
        combined_features = torch.cat([local_out, global_out], dim=-1)
        final_output = self.fusion(combined_features)
        
        # 注意力权重融合
        combined_weights = (local_weights + global_weights) / 2
        
        return final_output, combined_weights

class MeDeA(nn.Module):
    """Multi-disease Decompositional Attention 模型"""
    def __init__(self, num_classes, d_model=256, base_filters=64, dropout=0.3):
        super(MeDeA, self).__init__()      
        self.num_classes = num_classes
        
        # 共享骨干网络
        self.backbone = SharedBackbone(input_channels=12, base_filters=base_filters, dropout=dropout)
        
        # 为每个疾病创建专门的注意力头
        self.attention_heads = nn.ModuleList([
            AttentionHead(self.backbone.feature_dim, d_model, dropout)
            for _ in range(num_classes)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        pooled_features, feature_maps = self.backbone(x)
        
        outputs = []
        attention_weights = []
        
        # 为每个疾病类别计算输出
        for i, attention_head in enumerate(self.attention_heads):
            output, weights, _ = attention_head(pooled_features, feature_maps)
            outputs.append(output)
            attention_weights.append(weights)
        
        # 合并输出
        final_output = torch.cat(outputs, dim=1)  # (batch_size, num_classes)
        attention_weights = torch.stack(attention_weights, dim=1)  # (batch_size, num_classes, seq_len)
        
        return final_output, attention_weights

# ==============================================================================
# 4. 训练与评估逻辑 (Training & Evaluation)
# ==============================================================================
def run_evaluation(model, data_loader, device):
    """在验证集或测试集上运行评估。"""
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for signals, targets in data_loader:
            signals = signals.to(device, non_blocking=True)
            output = model(signals)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.sigmoid(output)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    preds = (all_probs > 0.5).astype(int)
    macro_f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
    
    try:
        macro_auc = roc_auc_score(all_targets, all_probs, average='macro')
    except ValueError:
        macro_auc = 0.0
        
    return macro_f1, macro_auc, all_targets, preds

def run_evaluation_detailed(model, data_loader, device, classes):
    """在验证集或测试集上运行详细评估，返回每个类别的性能指标。"""
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for signals, targets in data_loader:
            signals = signals.to(device, non_blocking=True)
            output = model(signals)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.sigmoid(output)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    preds = (all_probs > 0.5).astype(int)
    
    # 计算每个类别的性能指标
    per_class_metrics = {}
    
    for i, class_name in enumerate(classes):
        y_true_class = all_targets[:, i]
        y_pred_class = preds[:, i]
        y_prob_class = all_probs[:, i]
        
        # 计算 Precision, Recall, F1-Score
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y_true_class, y_pred_class, zero_division=0)
        recall = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
        
        # 计算 AUC
        try:
            auc = roc_auc_score(y_true_class, y_prob_class)
        except ValueError:
            auc = 0.0
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    
    # 计算宏平均
    macro_f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
    try:
        macro_auc = roc_auc_score(all_targets, all_probs, average='macro')
    except ValueError:
        macro_auc = 0.0
        
    return macro_f1, macro_auc, all_targets, preds, per_class_metrics

# ==============================================================================
# 5. 可解释性分析模块 (Explainability Analysis)
# ==============================================================================
def create_attention_overlay_plot(signal, attention_weights, classes, probabilities, 
                                true_labels, pred_labels, sample_idx, save_path):
    """
    创建注意力热力图叠加在ECG信号上的可视化
    """
    # 选择前3个导联进行显示
    signal_to_plot = signal[:3]  # (3, time_steps)
    signal_len = signal_to_plot.shape[1]
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 为每个类别创建一个子图
    num_classes = len(classes)
    
    for class_idx in range(num_classes):
        ax = plt.subplot(num_classes, 1, class_idx + 1)
        
        class_name = classes[class_idx]
        prob = probabilities[class_idx]
        true_label = true_labels[class_idx]
        pred_label = pred_labels[class_idx]
        
        # 获取该类别的注意力权重
        attention = attention_weights[class_idx]
        
        # 将注意力权重映射到信号长度
        attention_upsampled = np.interp(
            np.linspace(0, len(attention)-1, signal_len),
            np.arange(len(attention)),
            attention
        )
        
        # 平滑处理
        attention_smooth = gaussian_filter1d(attention_upsampled, sigma=2.0)
        
        # 归一化注意力权重到0-1
        if attention_smooth.max() > attention_smooth.min():
            attention_normalized = (attention_smooth - attention_smooth.min()) / (attention_smooth.max() - attention_smooth.min())
        else:
            attention_normalized = np.zeros_like(attention_smooth)
        
        # 绘制ECG信号（多导联叠加）
        for lead_idx in range(signal_to_plot.shape[0]):
            # 对每个导联添加垂直偏移以便区分
            offset = (lead_idx - 1) * 2  # 导联间距
            signal_with_offset = signal_to_plot[lead_idx] + offset
            
            # 绘制ECG基础信号
            ax.plot(signal_with_offset, color='black', alpha=0.8, linewidth=1.0, 
                   label=f'Lead {lead_idx+1}' if lead_idx < 3 else None)
        
        # 创建热力图背景
        # 使用注意力权重创建渐变背景
        x = np.arange(signal_len)
        y_min, y_max = ax.get_ylim()
        
        # 创建mesh grid用于热力图
        X, Y = np.meshgrid(x, np.linspace(y_min, y_max, 100))
        Z = np.tile(attention_normalized, (100, 1))
        
        # 根据预测概率选择颜色强度
        alpha_intensity = min(0.7, prob * 1.5)  # 概率越高，透明度越低（颜色越明显）
        
        # 选择颜色方案
        if true_label == 1:
            # 真实阳性：使用红色系
            cmap = plt.cm.Reds
        else:
            # 真实阴性：使用蓝色系
            cmap = plt.cm.Blues
            
        # 绘制热力图背景
        im = ax.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=alpha_intensity, 
                        vmin=0, vmax=1, extend='both')
        
        # 设置标题和标签
        status_symbol = "✓" if (true_label == pred_label) else "✗"
        status_color = "green" if (true_label == pred_label) else "red"
        
        title = f'{class_name} | P={prob:.3f} | True={int(true_label)} | Pred={int(pred_label)} {status_symbol}'
        ax.set_title(title, fontsize=12, fontweight='bold', 
                    color=status_color if true_label != pred_label else 'black')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # 只在第一个子图显示图例
        if class_idx == 0:
            ax.legend(loc='upper right')
    
    # 总标题
    plt.suptitle(f'Sample {sample_idx} - ECG Signal with Disease-Specific Attention Overlay', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 已保存注意力叠加图: {save_path}")

def create_combined_attention_overlay(signal, attention_weights, classes, probabilities, 
                                    true_labels, pred_labels, sample_idx, save_path):
    """
    创建所有类别注意力权重综合叠加的可视化
    只考虑预测概率较高的疾病类别
    """
    signal_to_plot = signal[:3]  # 前3个导联
    signal_len = signal_to_plot.shape[1]
    
    # 🔥 改进：只考虑预测概率较高的疾病类别
    # 设定阈值，过滤掉概率很低的类别
    prob_threshold = 0.5  # 可以调整这个阈值
    
    # 计算加权综合注意力 - 只包含有意义的疾病
    combined_attention = np.zeros(signal_len)
    active_diseases = []  # 记录参与计算的疾病
    
    print(f"🔍 样本 {sample_idx} 疾病概率分析:")
    for class_idx in range(len(classes)):
        class_name = classes[class_idx]
        prob = probabilities[class_idx]
        print(f"   - {class_name}: P={prob:.3f}")
        
        if prob > prob_threshold:  # 🔥 只考虑概率较高的疾病
            attention = attention_weights[class_idx]
            
            # 将注意力权重映射到信号长度
            attention_upsampled = np.interp(
                np.linspace(0, len(attention)-1, signal_len),
                np.arange(len(attention)),
                attention
            )
            
            # 平滑处理
            attention_smooth = gaussian_filter1d(attention_upsampled, sigma=2.0)
            
            # 根据预测概率加权
            combined_attention += attention_smooth * prob
            active_diseases.append(f"{class_name}(P={prob:.2f})")
            print(f"     ✓ 已纳入综合注意力计算")
        else:
            print(f"     ✗ 概率过低，已排除")
    
    # 归一化
    if combined_attention.max() > 0:
        combined_attention = combined_attention / combined_attention.max()
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    
    # 绘制ECG信号
    for lead_idx in range(signal_to_plot.shape[0]):
        offset = (lead_idx - 1) * 2
        signal_with_offset = signal_to_plot[lead_idx] + offset
        ax.plot(signal_with_offset, color='black', alpha=0.8, linewidth=1.5, 
               label=f'Lead {lead_idx+1}')
    
    # 创建热力图背景
    x = np.arange(signal_len)
    y_min, y_max = ax.get_ylim()
    
    X, Y = np.meshgrid(x, np.linspace(y_min, y_max, 100))
    Z = np.tile(combined_attention, (100, 1))
    
    # 🔥 改进：使用更具区分度的颜色映射
    im = ax.contourf(X, Y, Z, levels=50, cmap='plasma', alpha=0.6, vmin=0, vmax=1)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Combined Attention Weight (Active Diseases Only)', fontsize=12)
    
    # 🔥 改进：标题中显示参与计算的疾病
    active_diseases_str = ", ".join(active_diseases) if active_diseases else "No active diseases"
    title = f'Sample {sample_idx} - ECG Signal with Filtered Combined Attention\n'
    title += f'Active Diseases: {active_diseases_str}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 已保存过滤后的综合注意力叠加图: {save_path}")

def generate_class_comparison_plot(multi_label_samples, classes, explanations_dir):
    """
    生成类别间注意力比较图
    """
    if not multi_label_samples:
        print("⚠️ 没有样本可用于生成类别比较图")
        return
    
    print(f"🎨 正在生成类别间注意力比较图...")
    
    # 创建大图布局
    fig = plt.figure(figsize=(20, 12))
    
    # 为每个样本创建一个子图
    num_samples = min(3, len(multi_label_samples))
    
    for sample_idx in range(num_samples):
        sample = multi_label_samples[sample_idx]
        attention_weights = sample['attention_weights']  # (num_classes, seq_len)
        probabilities = sample['probabilities']
        true_labels = sample['true_labels']
        signal_len = attention_weights.shape[1]
        
        ax = plt.subplot(num_samples, 1, sample_idx + 1)
        
        # 获取活跃的疾病类别（概率 > 0.1）
        prob_threshold = 0.5
        active_indices = [i for i, prob in enumerate(probabilities) if prob > prob_threshold]
        
        if not active_indices:
            # 如果没有活跃疾病，选择概率最高的3个
            active_indices = np.argsort(probabilities)[-3:]
        
        # 为每个活跃疾病绘制注意力权重曲线
        colors = plt.cm.tab10(np.linspace(0, 1, len(active_indices)))
        
        for i, (class_idx, color) in enumerate(zip(active_indices, colors)):
            attention = attention_weights[class_idx]
            prob = probabilities[class_idx]
            true_label = true_labels[class_idx]
            class_name = classes[class_idx]
            
            # 平滑处理
            attention_smooth = gaussian_filter1d(attention, sigma=2.0)
            
            # 线条样式
            linestyle = '-' if true_label == 1 else '--'
            linewidth = 3 if prob > 0.5 else 2
            alpha = 0.9 if prob > 0.5 else 0.6
            
            label = f'{class_name} (P={prob:.3f}, True={int(true_label)})'
            
            ax.plot(attention_smooth, label=label, color=color, 
                   linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        
        # 设置子图标题和标签
        ax.set_title(f'Sample {sample_idx + 1} - Active Disease Attention Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Attention Weight')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, signal_len)
    
    plt.suptitle('Disease-Specific Attention Patterns Comparison Across Samples', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    save_path = explanations_dir / "class_attention_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 已保存类别间注意力比较图: {save_path}")

def generate_and_save_explanations(model, data_loader, classes, device, output_dir):
    """生成并保存多标签可解释性分析结果"""
    print("🔍 开始生成多标签可解释性分析...")
    
    model.eval()
    explanations_dir = output_dir / "explanations"
    explanations_dir.mkdir(exist_ok=True)
    
    # 收集多标签样本
    multi_label_samples = []
    
    with torch.no_grad():
        for signals, targets in data_loader:
            signals = signals.to(device, non_blocking=True)
            outputs, attention_weights = model(signals)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # 寻找具有多个标签的样本
            for batch_idx in range(signals.size(0)):
                true_labels = targets[batch_idx].cpu().numpy()
                pred_labels = preds[batch_idx].cpu().numpy()
                prob_values = probs[batch_idx].cpu().numpy()
                
                # 如果有多个真实标签或预测标签
                if np.sum(true_labels) > 1 or np.sum(pred_labels) > 1:
                    multi_label_samples.append({
                        'signal': signals[batch_idx].cpu().numpy(),
                        'attention_weights': attention_weights[batch_idx].cpu().numpy(),  # (num_classes, seq_len)
                        'true_labels': true_labels,
                        'pred_labels': pred_labels,
                        'probabilities': prob_values,
                        'num_true_labels': np.sum(true_labels),
                        'num_pred_labels': np.sum(pred_labels)
                    })
                
                # 收集足够样本后停止
                if len(multi_label_samples) >= 10:
                    break
            
            if len(multi_label_samples) >= 10:
                break
    
    if not multi_label_samples:
        print("⚠️ 没有找到多标签样本，生成单标签示例...")
        # 如果没有多标签样本，选择一些单标签样本
        with torch.no_grad():
            for signals, targets in data_loader:
                signals = signals.to(device, non_blocking=True)
                outputs, attention_weights = model(signals)
                probs = torch.sigmoid(outputs)
                
                for batch_idx in range(min(3, signals.size(0))):
                    multi_label_samples.append({
                        'signal': signals[batch_idx].cpu().numpy(),
                        'attention_weights': attention_weights[batch_idx].cpu().numpy(),
                        'true_labels': targets[batch_idx].cpu().numpy(),
                        'pred_labels': (probs[batch_idx] > 0.5).float().cpu().numpy(),
                        'probabilities': probs[batch_idx].cpu().numpy(),
                        'num_true_labels': np.sum(targets[batch_idx].cpu().numpy()),
                        'num_pred_labels': np.sum((probs[batch_idx] > 0.5).float().cpu().numpy())
                    })
                break

    # 选择最有代表性的样本（按标签数量排序）
    multi_label_samples.sort(key=lambda x: x['num_true_labels'], reverse=True)
    
    # 🔥 修改：生成改进的可解释性图
    for sample_idx, sample in enumerate(multi_label_samples[:3]):  # 选择前3个最复杂的样本
        print(f"🎨 正在生成样本 {sample_idx + 1} 的改进可解释性分析...")
        
        signal = sample['signal'][:3]  # 前3个导联
        attention_weights = sample['attention_weights']  # (num_classes, seq_len)
        true_labels = sample['true_labels']
        pred_labels = sample['pred_labels']
        probabilities = sample['probabilities']
        signal_len = signal.shape[1]
        
        # 🔥 识别确诊的疾病（真实标签为1的疾病）
        confirmed_diseases = [(i, classes[i]) for i, label in enumerate(true_labels) if label == 1]
        
        # 🔥 计算图的布局 - 根据确诊疾病数量动态调整
        num_confirmed = len(confirmed_diseases)
        if num_confirmed == 0:
            # 如果没有确诊疾病，选择概率最高的2个
            top_prob_indices = np.argsort(probabilities)[-2:]
            confirmed_diseases = [(i, classes[i]) for i in top_prob_indices]
            num_confirmed = len(confirmed_diseases)
        
        # 🔥 新的布局：ECG信号 + 各个确诊疾病热力图 + 组合热力图（去掉总结表格）
        total_rows = 1 + num_confirmed + 1  # ECG + 确诊疾病热力图 + 组合热力图
        
        # 🔥 调整图形大小和间距
        fig = plt.figure(figsize=(24, total_rows * 3.5))  # 增加每行高度
        gs = fig.add_gridspec(total_rows, 1, 
                             height_ratios=[1.2] + [0.8] * num_confirmed + [1.0],  # 调整高度比例
                             hspace=0.8)  # 🔥 增加垂直间距
        
        # 1. 原始ECG信号
        ax1 = fig.add_subplot(gs[0])
        for i in range(3):
            ax1.plot(signal[i], label=f'Lead {i+1}', alpha=0.8, linewidth=1.5)
        ax1.set_title(f'Sample {sample_idx + 1} - Original ECG Signal (First 3 Leads)', 
                     fontsize=18, fontweight='bold', pad=20)  # 增加标题间距
        ax1.set_xlabel('Time Steps', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, signal_len)
        
        # 2. 🔥 为每个确诊疾病单独绘制热力图 - 使用viridis颜色方案
        for heat_idx, (class_idx, class_name) in enumerate(confirmed_diseases):
            ax_heat = fig.add_subplot(gs[1 + heat_idx])
            
            # 获取该疾病的注意力权重
            attention = attention_weights[class_idx]
            prob = probabilities[class_idx]
            true_label = true_labels[class_idx]
            pred_label = pred_labels[class_idx]
            
            # 将注意力权重映射到信号长度
            attention_upsampled = np.interp(
                np.linspace(0, len(attention)-1, signal_len),
                np.arange(len(attention)),
                attention
            )
            
            # 平滑处理
            attention_smooth = gaussian_filter1d(attention_upsampled, sigma=1.5)
            
            # 🔥 使用viridis颜色方案（黄绿蓝）创建热力图
            attention_2d = attention_smooth.reshape(1, -1)
            
            im = ax_heat.imshow(attention_2d, cmap='viridis', aspect='auto', 
                               interpolation='bilinear', extent=[0, signal_len, 0.5, -0.5])
            
            # 设置状态指示
            if true_label == 1 and pred_label == 1:
                status = "TP ✓"
                title_color = "green"
            elif true_label == 1 and pred_label == 0:
                status = "FN ✗"
                title_color = "red"
            elif true_label == 0 and pred_label == 1:
                status = "FP ✗"
                title_color = "orange"
            else:
                status = "TN ✓"
                title_color = "blue"
            
            title = f'{class_name} Attention Heatmap | P={prob:.3f} | True={int(true_label)} | Pred={int(pred_label)} | {status}'
            ax_heat.set_title(title, fontsize=14, fontweight='bold', color=title_color, pad=15)  # 增加标题间距
            ax_heat.set_xlabel('Time Steps', fontsize=12)
            ax_heat.set_ylabel('Attention', fontsize=12)
            ax_heat.set_yticks([0])
            ax_heat.set_yticklabels(['Weight'])
            ax_heat.set_xlim(0, signal_len)
            
            # 🔥 调整colorbar位置和大小
            cbar = plt.colorbar(im, ax=ax_heat, orientation='horizontal', 
                               pad=0.4, shrink=0.6, aspect=30)  # 增加pad距离，缩小colorbar
            cbar.set_label('Attention Weight', fontsize=10)
        
        # 3. 🔥 组合热力图 - 拉长到与疾病热力图相同长度
        ax_combined = fig.add_subplot(gs[1 + num_confirmed])
        
        # 计算加权综合注意力 - 只包含确诊疾病
        combined_attention = np.zeros(signal_len)
        active_diseases_info = []
        
        for class_idx, class_name in confirmed_diseases:
            prob = probabilities[class_idx]
            attention = attention_weights[class_idx]
            
            # 将注意力权重映射到信号长度
            attention_upsampled = np.interp(
                np.linspace(0, len(attention)-1, signal_len),
                np.arange(len(attention)),
                attention
            )
            
            # 根据预测概率加权
            combined_attention += gaussian_filter1d(attention_upsampled, sigma=2.0) * prob
            active_diseases_info.append(f"{class_name}(P={prob:.2f})")
        
        # 归一化
        if combined_attention.max() > 0:
            combined_attention = combined_attention / combined_attention.max()
        
        # 🔥 绘制拉长的组合热力图 - 使用viridis颜色方案
        combined_2d = combined_attention.reshape(1, -1)
        im_combined = ax_combined.imshow(combined_2d, cmap='viridis', aspect='auto', 
                                        interpolation='bilinear', extent=[0, signal_len, 0.5, -0.5])
        
        # 设置标题
        combined_title = f'Probability-Weighted Combined Attention (Confirmed Diseases)\nActive: {", ".join(active_diseases_info)}'
        ax_combined.set_title(combined_title, fontsize=16, fontweight='bold', pad=20)  # 增加标题间距
        ax_combined.set_xlabel('Time Steps', fontsize=14)
        ax_combined.set_ylabel('Combined\nAttention', fontsize=12)
        ax_combined.set_yticks([0])
        ax_combined.set_yticklabels(['Weighted'])
        ax_combined.set_xlim(0, signal_len)
        
        # 🔥 调整colorbar位置和大小
        cbar_combined = plt.colorbar(im_combined, ax=ax_combined, orientation='horizontal', 
                                    pad=0.3, shrink=0.6, aspect=30)  # 增加pad距离，缩小colorbar
        cbar_combined.set_label('Combined Attention Weight', fontsize=12)
        
        # 🔥 设置总标题 - 简化信息
        confirmed_names = [name for _, name in confirmed_diseases]
        plt.suptitle(f'Sample {sample_idx + 1} - Disease-Specific Attention Analysis\n'
                    f'Confirmed Diseases: {", ".join(confirmed_names)} | '
                    f'Correctly Predicted: {np.sum(true_labels == pred_labels)}/{len(classes)}', 
                    fontsize=20, fontweight='bold', y=0.98)  # 调整总标题位置
        
        # 保存改进的可解释性图像
        save_path = explanations_dir / f"enhanced_explanation_sample_{sample_idx + 1}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   pad_inches=0.5)  # 增加边距
        plt.close()
        
        print(f"✅ 已保存样本 {sample_idx + 1} 的改进可解释性分析图: {save_path}")
        
        # 🔥 继续生成其他类型的图（保持原有功能）
        print(f"🎨 正在生成样本 {sample_idx + 1} 的其他可解释性图...")
        
        # 个体类别注意力叠加图
        overlay_individual_path = explanations_dir / f"attention_overlay_individual_sample_{sample_idx + 1}.png"
        create_attention_overlay_plot(
            signal=sample['signal'],
            attention_weights=sample['attention_weights'],
            classes=classes,
            probabilities=sample['probabilities'],
            true_labels=sample['true_labels'],
            pred_labels=sample['pred_labels'],
            sample_idx=sample_idx + 1,
            save_path=overlay_individual_path
        )
        
        # 综合注意力叠加图
        overlay_combined_path = explanations_dir / f"attention_overlay_combined_sample_{sample_idx + 1}.png"
        create_combined_attention_overlay(
            signal=sample['signal'],
            attention_weights=sample['attention_weights'],
            classes=classes,
            probabilities=sample['probabilities'],
            true_labels=sample['true_labels'],
            pred_labels=sample['pred_labels'],
            sample_idx=sample_idx + 1,
            save_path=overlay_combined_path
        )
    
    # 生成类别间注意力比较图
    print("🎨 正在生成类别间注意力比较图...")
    generate_class_comparison_plot(multi_label_samples[:5], classes, explanations_dir)
    
    print(f"🎉 改进的多标签可解释性分析完成！所有图像已保存到: {explanations_dir}")
    print(f"   - 🔥 改进的可解释性图: enhanced_explanation_sample_*.png")
    print(f"   - 个体注意力叠加图: attention_overlay_individual_sample_*.png")
    print(f"   - 综合注意力叠加图: attention_overlay_combined_sample_*.png")
    print(f"   - 类别比较图: class_attention_comparison.png")
    print(f"\n🎯 主要改进:")
    print(f"   ✓ 确诊疾病热力图分开绘制，使用统一的viridis颜色方案（黄绿蓝）")
    print(f"   ✓ 组合热力图长度与疾病热力图保持一致")
    print(f"   ✓ 去掉总结表格，避免文字重叠")
    print(f"   ✓ 增加图间距和标题间距，改善视觉效果")

def plot_confusion_matrices(y_true, y_pred, classes, output_dir):
    """为每个类别绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    
    output_dir = Path(output_dir)
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    
    print(f"🎨 正在生成混淆矩阵...")
    
    # 为每个类别单独绘制混淆矩阵
    for i, class_name in enumerate(classes):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true_class, y_pred_class)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {class_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # 保存图像
        save_path = cm_dir / f"confusion_matrix_{class_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 🔥 绘制综合混淆矩阵（所有类别的总体表现）
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # 假设有5个类别
    axes = axes.flatten()
    
    for i, (class_name, ax) in enumerate(zip(classes, axes)):
        if i >= len(classes):
            ax.axis('off')
            continue
        
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        cm = confusion_matrix(y_true_class, y_pred_class)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Neg', 'Pos'],
                   yticklabels=['Neg', 'Pos'])
        ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True', fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10)
    
    # 隐藏多余的子图
    for j in range(len(classes), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Confusion Matrices for All Classes', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    combined_path = cm_dir / "confusion_matrices_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 混淆矩阵已保存到: {cm_dir}")
    print(f"   - 单个类别: confusion_matrix_<class_name>.png")
    print(f"   - 综合视图: confusion_matrices_combined.png")

def plot_multilabel_confusion_matrix(y_true, y_pred, classes, output_dir):
    """
    为多标签分类生成共现混淆矩阵
    显示：当真实标签为类别i时，预测为类别j的比例
    """
    from sklearn.metrics import multilabel_confusion_matrix
    import pandas as pd
    
    output_dir = Path(output_dir)
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    
    print(f"🎨 正在生成多标签共现混淆矩阵...")
    
    # 🔥 方法1：预测率矩阵（Prediction Rate Matrix）
    # 计算：当真实标签为i时，预测为j的样本占真实为i的样本的比例
    n_classes = len(classes)
    prediction_rate_matrix = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        # 找到真实标签为类别i的样本
        true_class_i_mask = y_true[:, i] == 1
        n_true_class_i = true_class_i_mask.sum()
        
        if n_true_class_i > 0:
            for j in range(n_classes):
                # 在这些样本中，预测为类别j的比例
                pred_class_j_in_true_i = (y_pred[true_class_i_mask, j] == 1).sum()
                prediction_rate_matrix[i, j] = pred_class_j_in_true_i / n_true_class_i
    
    # 绘制预测率矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(prediction_rate_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes,
                cbar_kws={'label': 'Prediction Rate'},
                vmin=0, 
                vmax=1)
    
    plt.title('Multi-Label Prediction Rate Matrix\n(Proportion of samples where Pred=1 | True=1)', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Condition', fontsize=14)
    plt.ylabel('True Condition Present', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = cm_dir / "multilabel_prediction_rate_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 预测率矩阵已保存: {save_path}")
    
    # 🔥 方法2：共现计数矩阵（Co-occurrence Count Matrix）
    # 计算：真实为i且预测为j的样本数量
    cooccurrence_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(n_classes):
        for j in range(n_classes):
            # 真实标签为i且预测标签为j的样本数
            cooccurrence_matrix[i, j] = ((y_true[:, i] == 1) & (y_pred[:, j] == 1)).sum()
    
    # 绘制共现计数矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence_matrix, 
                annot=True, 
                fmt='d', 
                cmap='YlOrRd',
                xticklabels=classes,
                yticklabels=classes,
                cbar_kws={'label': 'Sample Count'})
    
    plt.title('Co-occurrence Count Matrix\n(Number of samples where True=1 AND Pred=1)', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Condition', fontsize=14)
    plt.ylabel('True Condition Present', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = cm_dir / "multilabel_cooccurrence_count_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 共现计数矩阵已保存: {save_path}")
    
    # 🔥 方法3：样式化的条件概率矩阵（更接近图2的风格）
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # 使用更强的对比度颜色方案
    im = ax.imshow(prediction_rate_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个单元格中显示数值
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, f'{prediction_rate_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if prediction_rate_matrix[i, j] > 0.5 else "black",
                          fontsize=12, fontweight='bold')
    
    # 添加标题和标签
    ax.set_title('Multi-Label Confusion Matrix\n(Prediction Rate: P(Pred=j | True=i))', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Condition Present', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prediction Rate', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = cm_dir / "multilabel_confusion_matrix_styled.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 样式化混淆矩阵已保存: {save_path}")
    
    # 🔥 保存数值矩阵为CSV（方便后续分析）
    df_prediction_rate = pd.DataFrame(
        prediction_rate_matrix, 
        index=[f"True_{cls}" for cls in classes],
        columns=[f"Pred_{cls}" for cls in classes]
    )
    df_prediction_rate.to_csv(cm_dir / "prediction_rate_matrix.csv")
    
    df_cooccurrence = pd.DataFrame(
        cooccurrence_matrix, 
        index=[f"True_{cls}" for cls in classes],
        columns=[f"Pred_{cls}" for cls in classes]
    )
    df_cooccurrence.to_csv(cm_dir / "cooccurrence_count_matrix.csv")
    
    print(f"✅ 数值矩阵已保存为CSV文件")
    print(f"\n📊 多标签混淆矩阵生成完成！")
    print(f"   - 预测率矩阵: multilabel_prediction_rate_matrix.png")
    print(f"   - 共现计数矩阵: multilabel_cooccurrence_count_matrix.png")
    print(f"   - 样式化矩阵: multilabel_confusion_matrix_styled.png")
    print(f"   - CSV数据: prediction_rate_matrix.csv, cooccurrence_count_matrix.csv")
    
    return prediction_rate_matrix, cooccurrence_matrix

# 在交叉验证中调用
def run_cross_validation(args):
    """运行十折交叉验证。"""
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 MeDeA 模型十折交叉验证...")
    print(f"   - 设备: {device}")
    print(f"   - 结果将保存至: {output_dir}")
    
    # 存储所有fold的结果
    all_fold_results = {
        'val_f1': [],
        'val_auc': [],
        'test_f1': [],
        'test_auc': [],
        'fold_times': [],
        'per_class_metrics': []  # 🔥 新增：存储每个fold的每类别指标
    }
    
    classes = None  # 用于存储类别名称
    
    # 运行十折交叉验证
    for fold in range(1, 11):  # fold1 到 fold10
        print(f"\n{'='*60}")
        print(f"🔄 开始第 {fold}/10 折交叉验证")
        print(f"{'='*60}")
        
        fold_start_time = time.time()
        
        # 为当前fold创建专用输出目录
        fold_output_dir = output_dir / f"fold_{fold}"
        fold_output_dir.mkdir(exist_ok=True)
        
        # 加载当前fold的数据
        try:
            train_loader, val_loader, test_loader, fold_classes = create_dataloaders_cv(
                args.data_dir, fold, args.batch_size, args.num_workers
            )
            if classes is None:
                classes = fold_classes  # 保存类别名称
        except FileNotFoundError as e:
            print(f"⚠️ 跳过第 {fold} 折: {e}")
            continue
        
        # 创建模型
        model = MeDeA(
            num_classes=len(classes),
            d_model=args.d_model,
            base_filters=args.base_filters,
            dropout=args.dropout
        ).to(device)
        
        if fold == 1:  # 只在第一折显示模型信息
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"🏥 MeDeA 模型配置: d_model={args.d_model}, base_filters={args.base_filters}")
            print(f"   - 可训练参数: {param_count:,}")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        
        best_f1, patience_counter = 0.0, 0
        
        # 训练循环
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Fold {fold} Epoch {epoch+1}/{args.epochs}')
            
            for signals, targets in progress_bar:
                signals, targets = signals.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                outputs, attention_weights = model(signals)
                
                # 主要分类损失
                classification_loss = criterion(outputs, targets)
                
                # 注意力分布正则化损失
                attention_entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-8), 
                    dim=-1
                ).mean()
                
                # 组合损失
                total_loss_batch = classification_loss - 0.01 * attention_entropy
                
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                progress_bar.set_postfix({
                    'cls_loss': f'{classification_loss.item():.4f}',
                    'att_entropy': f'{attention_entropy.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

            val_f1, val_auc, _, _, _ = run_evaluation_detailed(model, val_loader, device, classes)
            scheduler.step(val_f1)
            
            if args.verbose or epoch % 10 == 0:  # 每10个epoch或verbose模式下显示
                print(f"Fold {fold} Epoch {epoch+1}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Loss: {total_loss / len(train_loader):.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # 保存当前fold的最佳模型
                torch.save(model.state_dict(), fold_output_dir / 'best_medea_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"⌛️ Fold {fold} 早停: 在第 {epoch+1} 个epoch触发")
                break
        
        # 🔥 在测试集上评估最终模型 - 获取详细指标
        model.load_state_dict(torch.load(fold_output_dir / 'best_medea_model.pth', map_location=device))
        test_f1, test_auc, test_true, test_preds, per_class_metrics = run_evaluation_detailed(model, test_loader, device, classes)
        
        # 保存预测结果
        np.savez(
            fold_output_dir / 'predictions.npz',
            y_true=test_true,
            y_pred=test_preds,
            classes=classes
        )
        
        # 🔥 生成混淆矩阵
        plot_confusion_matrices(test_true, test_preds, classes, fold_output_dir)
        
        # 🔥 生成多标签共现混淆矩阵
        plot_multilabel_confusion_matrix(test_true, test_preds, classes, fold_output_dir)
        
        # 🔥 新增：保存预测结果和真实标签
        np.savez(
            fold_output_dir / 'predictions.npz',
            y_true=test_true,           # 真实标签 (n_samples, n_classes)
            y_pred=test_preds,          # 预测标签 (n_samples, n_classes) - 二值化后
            classes=classes             # 类别名称
        )
        print(f"✅ Fold {fold} 预测结果已保存到: {fold_output_dir / 'predictions.npz'}")
        
        fold_time = time.time() - fold_start_time
        
        # 保存当前fold的结果
        all_fold_results['val_f1'].append(best_f1)
        all_fold_results['val_auc'].append(val_auc)
        all_fold_results['test_f1'].append(test_f1)
        all_fold_results['test_auc'].append(test_auc)
        all_fold_results['fold_times'].append(fold_time)
        all_fold_results['per_class_metrics'].append(per_class_metrics)  # 🔥 保存每类别指标
        
        print(f"\n📊 第 {fold} 折结果:")
        print(f"   - 最佳验证 F1: {best_f1:.4f}")
        print(f"   - 测试 F1: {test_f1:.4f}")
        print(f"   - 测试 AUC: {test_auc:.4f}")
        print(f"   - 用时: {fold_time/60:.2f} 分钟")
        
        # 保存详细的分类报告
        with open(fold_output_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Fold {fold} Classification Report\n")
            f.write("="*50 + "\n")
            f.write(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
            f.write(f"\nTest F1-Score: {test_f1:.4f}\n")
            f.write(f"Test AUC: {test_auc:.4f}\n")
            
            # 🔥 添加每个类别的详细指标
            f.write(f"\nPer-Class Performance:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}\n")
            f.write("-"*50 + "\n")
            for class_name, metrics in per_class_metrics.items():
                f.write(f"{class_name:<8} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1_score']:<10.4f} {metrics['auc']:<10.4f}\n")
        
        # 如果需要生成可解释性分析（仅在指定fold）
        if args.run_explain and fold in args.explain_folds:
            print(f"🔍 为第 {fold} 折生成可解释性分析...")
            generate_and_save_explanations(
                model=model,
                data_loader=test_loader,
                classes=classes,
                device=device,
                output_dir=fold_output_dir
            )
    
    # 🔥 计算并保存详细的交叉验证总结果
    if all_fold_results['test_f1'] and classes is not None:
        print(f"\n{'='*60}")
        print("🎉 十折交叉验证完成！")
        print(f"{'='*60}")
        
        # 计算宏平均统计
        mean_val_f1 = np.mean(all_fold_results['val_f1'])
        std_val_f1 = np.std(all_fold_results['val_f1'])
        mean_test_f1 = np.mean(all_fold_results['test_f1'])
        std_test_f1 = np.std(all_fold_results['test_f1'])
        mean_test_auc = np.mean(all_fold_results['test_auc'])
        std_test_auc = np.std(all_fold_results['test_auc'])
        total_time = sum(all_fold_results['fold_times'])
        
        # 🔥 计算每个类别的平均性能指标
        per_class_summary = {}
        for class_name in classes:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            auc_scores = []
            
            for fold_metrics in all_fold_results['per_class_metrics']:
                if class_name in fold_metrics:
                    precision_scores.append(fold_metrics[class_name]['precision'])
                    recall_scores.append(fold_metrics[class_name]['recall'])
                    f1_scores.append(fold_metrics[class_name]['f1_score'])
                    auc_scores.append(fold_metrics[class_name]['auc'])
            
            per_class_summary[class_name] = {
                'precision_mean': np.mean(precision_scores),
                'precision_std': np.std(precision_scores),
                'recall_mean': np.mean(recall_scores),
                'recall_std': np.std(recall_scores),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'auc_mean': np.mean(auc_scores),
                'auc_std': np.std(auc_scores)
            }
        
        print(f"📈 交叉验证结果总结:")
        print(f"   - 验证 F1-Score: {mean_val_f1:.4f} ± {std_val_f1:.4f}")
        print(f"   - 测试 F1-Score: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"   - 测试 AUC:      {mean_test_auc:.4f} ± {std_test_auc:.4f}")
        print(f"   - 总用时: {total_time/60:.2f} 分钟")
        
        # 🔥 打印每个类别的平均性能
        print(f"\n📊 Per-Class Performance Summary:")
        print(f"{'Class':<8} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'AUC':<15}")
        print("-" * 75)
        for class_name, metrics in per_class_summary.items():
            precision_str = f"{metrics['precision_mean']:.3f}±{metrics['precision_std']:.3f}"
            recall_str = f"{metrics['recall_mean']:.3f}±{metrics['recall_std']:.3f}"
            f1_str = f"{metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f}"
            auc_str = f"{metrics['auc_mean']:.3f}±{metrics['auc_std']:.3f}"
            
            print(f"{class_name:<8} {precision_str:<15} {recall_str:<15} {f1_str:<15} {auc_str:<15}")
        
        # 保存交叉验证结果总结
        results_summary = {
            'cross_validation_results': all_fold_results,
            'summary_statistics': {
                'mean_val_f1': mean_val_f1,
                'std_val_f1': std_val_f1,
                'mean_test_f1': mean_test_f1,
                'std_test_f1': std_test_f1,
                'mean_test_auc': mean_test_auc,
                'std_test_auc': std_test_auc,
                'total_time_minutes': total_time/60
            },
            'per_class_summary': per_class_summary  # 🔥 添加每类别总结
        }
        
        # 保存为JSON文件
        import json
        with open(output_dir / 'cross_validation_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 🔥 保存详细的文本报告
        with open(output_dir / 'cross_validation_report.txt', 'w') as f:
            f.write("MeDeA Model - 10-Fold Cross Validation Results\n")
            f.write("="*80 + "\n\n")
            
            # 总体性能统计
            f.write("Overall Performance Summary:\n")
            f.write("-"*40 + "\n")
            f.write(f"Validation F1-Score: {mean_val_f1:.4f} ± {std_val_f1:.4f}\n")
            f.write(f"Test F1-Score:       {mean_test_f1:.4f} ± {std_test_f1:.4f}\n")
            f.write(f"Test AUC:            {mean_test_auc:.4f} ± {std_test_auc:.4f}\n")
            f.write(f"Total Time:          {total_time/60:.2f} minutes\n\n")
            
            # 🔥 Per-Class Performance Table (像你要求的表格格式)
            f.write("TABLE III\n")
            f.write("PER-CLASS PERFORMANCE OF MEDEA (MULTI-QUERY).\n")
            f.write("="*80 + "\n")
            f.write(f"{'Class':<8} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'AUC':<15}\n")
            f.write("-"*80 + "\n")
            
            for class_name, metrics in per_class_summary.items():
                precision_str = f"{metrics['precision_mean']*100:.2f}%"
                recall_str = f"{metrics['recall_mean']*100:.2f}%"
                f1_str = f"{metrics['f1_mean']*100:.2f}%"
                auc_str = f"{metrics['auc_mean']*100:.2f}%"
                
                f.write(f"{class_name:<8} {precision_str:<15} {recall_str:<15} {f1_str:<15} {auc_str:<15}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            
            # 🔥 详细的Individual Fold Results
            f.write("Individual Fold Results:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Fold':<6} {'Val F1':<8} {'Test F1':<8} {'Test AUC':<8} {'Time(min)':<10}\n")
            f.write("-"*80 + "\n")
            for i, (vf1, tf1, tauc, t) in enumerate(zip(
                all_fold_results['val_f1'], all_fold_results['test_f1'], 
                all_fold_results['test_auc'], all_fold_results['fold_times']
            ), 1):
                f.write(f"Fold {i:<2} {vf1:<8.4f} {tf1:<8.4f} {tauc:<8.4f} {t/60:<10.1f}\n")
            
            f.write("\n" + "-"*80 + "\n\n")
            
            # 🔥 Per-Class Performance Across All Folds (详细版本)
            f.write("Detailed Per-Class Performance Across All Folds:\n")
            f.write("-"*80 + "\n")
            
            for class_name in classes:
                f.write(f"\n{class_name} Performance:\n")
                f.write(f"{'Fold':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}\n")
                f.write("-"*50 + "\n")
                
                for i, fold_metrics in enumerate(all_fold_results['per_class_metrics'], 1):
                    if class_name in fold_metrics:
                        metrics = fold_metrics[class_name]
                        f.write(f"Fold {i:<2} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                               f"{metrics['f1_score']:<10.4f} {metrics['auc']:<10.4f}\n")
                
                # 类别统计
                class_stats = per_class_summary[class_name]
                f.write("-"*50 + "\n")
                f.write(f"Mean:  {class_stats['precision_mean']:<10.4f} {class_stats['recall_mean']:<10.4f} "
                       f"{class_stats['f1_mean']:<10.4f} {class_stats['auc_mean']:<10.4f}\n")
                f.write(f"Std:   {class_stats['precision_std']:<10.4f} {class_stats['recall_std']:<10.4f} "
                       f"{class_stats['f1_std']:<10.4f} {class_stats['auc_std']:<10.4f}\n")
        
        print(f"📁 所有结果已保存到: {output_dir}")
        print(f"📊 详细的Per-Class Performance表格已保存到: {output_dir}/cross_validation_report.txt")
    else:
        print("❌ 没有成功完成任何fold的训练")

def main_training_single_fold(args):
    """训练单个指定的fold。"""
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 MeDeA 模型训练 (Fold {args.fold})...")
    print(f"   - 设备: {device}")
    print(f"   - 模型将保存至: {output_dir}")
    
    train_loader, val_loader, test_loader, classes = create_dataloaders_cv(
        args.data_dir, args.fold, args.batch_size, args.num_workers
    )
    
    model = MeDeA(
        num_classes=len(classes),
        d_model=args.d_model,
        base_filters=args.base_filters,
        dropout=args.dropout
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏥 MeDeA 模型配置: d_model={args.d_model}, base_filters={args.base_filters}")
    print(f"   - 可训练参数: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    best_f1, patience_counter = 0.0, 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for signals, targets in progress_bar:
            signals, targets = signals.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            outputs, attention_weights = model(signals)
            
            # 主要分类损失
            classification_loss = criterion(outputs, targets)
            
            # 注意力分布正则化损失
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8), 
                dim=-1
            ).mean()
            
            # 组合损失
            total_loss_batch = classification_loss - 0.01 * attention_entropy
            
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            progress_bar.set_postfix({
                'cls_loss': f'{classification_loss.item():.4f}',
                'att_entropy': f'{attention_entropy.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        val_f1, _, _, _ = run_evaluation(model, val_loader, device)
        scheduler.step(val_f1)
        print(f"Epoch {epoch+1}, Val F1-macro: {val_f1:.4f}, Avg Loss: {total_loss / len(train_loader):.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            print(f"📈 新的最佳验证 F1 分数: {best_f1:.4f}. 模型已保存。")
            torch.save(model.state_dict(), output_dir / 'best_medea_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"⌛️ 早停: 在 {epoch+1} 个epoch触发")
            break
            
    total_time = time.time() - start_time
    print(f"\n✅ 训练完成，总耗时: {total_time/60:.2f} 分钟。")

    print(f"\n🚀 在测试集上评估最终模型 (Fold {args.fold})...")
    model.load_state_dict(torch.load(output_dir / 'best_medea_model.pth', map_location=device))
    test_f1, test_auc, test_true, test_preds = run_evaluation(model, test_loader, device)

    print(f"\n--- 🚀 MeDeA 模型最终分类报告 (Fold {args.fold}) ---")
    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
    print(f"\n--- 🚀 Test Set Performance ---")
    print(f"Macro F1-Score: {test_f1:.4f}")
    print(f"Macro AUC:      {test_auc:.4f}")

    # 🔥 新增：保存预测结果
    np.savez(
        output_dir / 'predictions.npz',
        y_true=test_true,
        y_pred=test_preds,
        classes=classes
    )
    print(f"✅ 预测结果已保存到: {output_dir / 'predictions.npz'}")
    
    # 🔥 新增：生成混淆矩阵
    print(f"\n📊 正在生成混淆矩阵...")
    plot_confusion_matrices(test_true, test_preds, classes, output_dir)
    
    # 🔥 生成多标签共现混淆矩阵
    print(f"\n📊 正在生成多标签共现混淆矩阵...")
    plot_multilabel_confusion_matrix(test_true, test_preds, classes, output_dir)

    # 可解释性分析（可选）
    if args.run_explain:
        generate_and_save_explanations(
            model=model,
            data_loader=test_loader,
            classes=classes,
            device=device,
            output_dir=output_dir
        )
    else:
        print("\n⏭️  跳过可解释性分析。如需生成热力图，请在运行时加入 --run_explain 参数。")

# 修改 run_explain_only 函数以支持fold参数
def run_explain_only(args):
    """仅运行可解释性分析，不进行训练。"""
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    
    # 检查模型文件是否存在
    model_path = output_dir / 'best_medea_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 找不到已训练的模型文件: {model_path}")
    
    print(f"🔍 启动可解释性分析模式 (Fold {args.fold})...")
    print(f"   - 设备: {device}")
    print(f"   - 模型路径: {model_path}")
    
    # 加载数据 - 使用交叉验证数据加载器
    _, _, test_loader, classes = create_dataloaders_cv(
        args.data_dir, args.fold, args.batch_size, args.num_workers
    )
    

    
    # 创建模型并加载权重
    model = MeDeA(
        num_classes=len(classes),
        d_model=args.d_model,
        base_filters=args.base_filters,
        dropout=args.dropout
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ 模型权重加载完成")
    

    
    # 运行可解释性分析
    generate_and_save_explanations(
        model=model,
        data_loader=test_loader,
        classes=classes,
        device=device,
        output_dir=output_dir
    )

# ==============================================================================
# 7. 主程序入口 (Main Entry)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🚀 MeDeA 模型训练与可解释性分析脚本')
    
    # 🔥 修改数据路径参数
    parser.add_argument('--data_dir', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data', help='包含十折交叉验证数据的目录路径')
    parser.add_argument('--output_dir', type=str, default='./saved_models/medea_experiment', help='保存模型和结果的目录')
    
    # 🔥 添加交叉验证相关参数
    parser.add_argument('--cross_validation', action='store_true', help='运行十折交叉验证')
    parser.add_argument('--fold', type=int, default=1, help='指定训练单个fold (1-10)')
    parser.add_argument('--explain_folds', type=int, nargs='+', default=[1], help='为哪些fold生成可解释性分析 (例如: --explain_folds 1 5  10)')
    parser.add_argument('--verbose', action='store_true', help='显示详细的训练过程')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--d_model', type=int, default=256, help='注意力头的隐藏维度')
    parser.add_argument('--base_filters', type=int, default=64, help='CNN骨干网络的基础滤波器数量')
    parser.add_argument('--dropout', type=float, default=0.3)
    
    parser.add_argument('--run_explain', action='store_true', help='在训练结束后运行可解释性分析并生成热力图')
    parser.add_argument('--explain_only', action='store_true', help='仅运行可解释性分析，不进行训练（需要已存在的模型文件）')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    
    args = parser.parse_args()
    
    if args.explain_only:
        # 🔥 修改explain_only模式以支持fold
        args.fold = args.fold if hasattr(args, 'fold') else 1
        run_explain_only(args)
    elif args.cross_validation:
        # 🔥 运行十折交叉验证
        run_cross_validation(args)
    else:
        # 🔥 训练单个fold
        main_training_single_fold(args)