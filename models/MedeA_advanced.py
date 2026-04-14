# -*- coding: utf-8 -*-
"""
MeDeA-Pro: Advanced Multi-disease Decompositional Attention
[架构升级版]：
1. Backbone: 升级为 SE-ResNet1D (Squeeze-and-Excitation)，大幅增强特征提取能力。
2. Attention: 保持 Multi-Query Attention (专家委员会) 机制。
3. Training: 优化了 Entropy 正则化的计算路径。

此脚本是一个独立的、可直接运行的文件，包含了完整的数据加载、训练、评估和可视化流程。
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
    print(f"✅ [Advanced] 所有随机种子已设置为: {seed}")

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

def create_dataloaders_cv(data_dir, fold_num, batch_size, num_workers):
    """从十折交叉验证数据文件夹加载指定fold的数据并创建DataLoaders。"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"❌ 数据目录未找到: {data_dir}")
    
    data_file_path = data_dir / f"ptbxl_processed_100hz_fold{fold_num}.npz"
    
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
    print(f"    - 训练集: {len(train_ds)} 样本")
    print(f"    - 验证集: {len(val_ds)} 样本") 
    print(f"    - 测试集: {len(test_ds)} 样本")
    
    return train_loader, val_loader, test_loader, classes

# ==============================================================================
# 3. [核心修改] 高级骨干网络: SE-ResNet1D (Squeeze-and-Excitation)
# ==============================================================================

class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D signals.
    通过建模通道间的依赖关系，自适应地重新校准通道特征响应。
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResNetBasicBlock1d(nn.Module):
    """带有 SE 模块的 ResNet BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=7, dropout=0.1):
        super(ResNetBasicBlock1d, self).__init__()
        # 使用更大的 kernel_size (如7) 通常对 ECG 这种时序信号更好，捕捉形态学特征
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        # 嵌入 SE Block
        self.se = SEBlock1d(planes, reduction=8) 
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE Attention
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AdvancedSEResNet1d(nn.Module):
    """
    高级 SE-ResNet1D 骨干网络。
    结构更深，感受野更大，适合提取复杂的病理特征。
    """
    def __init__(self, input_channels=12, layers=[3, 4, 6, 3], base_filters=64, kernel_size=7, dropout=0.2):
        super(AdvancedSEResNet1d, self).__init__()
        self.inplanes = base_filters
        
        # 初始层：大卷积核快速下采样，捕捉低频特征
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 堆叠 ResNet 层
        self.layer1 = self._make_layer(ResNetBasicBlock1d, base_filters, layers[0], stride=1, kernel_size=kernel_size, dropout=dropout)
        self.layer2 = self._make_layer(ResNetBasicBlock1d, base_filters * 2, layers[1], stride=2, kernel_size=kernel_size, dropout=dropout)
        self.layer3 = self._make_layer(ResNetBasicBlock1d, base_filters * 4, layers[2], stride=2, kernel_size=kernel_size, dropout=dropout)
        self.layer4 = self._make_layer(ResNetBasicBlock1d, base_filters * 8, layers[3], stride=2, kernel_size=kernel_size, dropout=dropout)

        # 最终特征维度
        self.feature_dim = base_filters * 8 * ResNetBasicBlock1d.expansion

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=7, dropout=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel_size, dropout=dropout))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size, dropout=dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 12, L)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_maps = self.layer4(x) # (B, feature_dim, L')
        
        return feature_maps

# ==============================================================================
# 4. Multi-Query Attention Head (专家委员会机制)
# ==============================================================================
class MultiQueryAttentionHead(nn.Module):
    """
    真正的 Multi-Query Attention Head
    每个头内部由 H (num_queries) 个 learnable query vectors 组成“专家委员会”。
    """
    def __init__(self, feature_dim, d_model, num_queries=4, dropout=0.1):
        super(MultiQueryAttentionHead, self).__init__()
        self.d_model = d_model
        self.num_queries = num_queries  # H: 委员会规模
        
        # Key & Value 映射
        self.key = nn.Linear(feature_dim, d_model)
        self.value = nn.Linear(feature_dim, d_model)
        
        # 可学习的 Query 向量组 (Expert Committee)
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model * num_queries)
        
        # 分类器 (Concatenation后维度变大)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model), # 降维融合
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, feature_maps):
        # feature_maps: (batch_size, feature_dim, seq_len)
        batch_size = feature_maps.size(0)
        
        # (B, L, F)
        feature_seq = feature_maps.transpose(1, 2)
        
        K = self.key(feature_seq)   # (B, L, D)
        V = self.value(feature_seq) # (B, L, D)
        
        # Q: (B, H, D)
        Q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attention Scores: (B, H, L)
        # 这里的 H 个 channel 代表 H 个不同的查询生成的 H 张热力图
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Attention Probs
        attn_probs = F.softmax(attention_scores, dim=-1) # (B, H, L)
        
        # 🔥 [High Performance Optimization] 计算精确的 Entropy
        # 计算每个样本、每个Query的熵，然后取平均，用于返回给主模型做正则化
        # Entropy = - sum(p * log(p))
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1).mean()
        
        # Context: (B, H, D)
        context_vectors = torch.matmul(attn_probs, V)
        
        # 拼接 (Concatenation)
        # 将 H 个上下文向量展平成一个长向量: (B, H*D)
        concat_context = context_vectors.view(batch_size, -1)
        
        concat_context = self.layer_norm(concat_context)
        concat_context = self.dropout(concat_context)
        
        # 分类
        output = self.classifier(concat_context)
        
        # 为了可视化，我们需要返回一个 (B, L) 的权重
        # 论文方法：对 H 个 Query 的权重取平均
        avg_weights = attn_probs.mean(dim=1)
        
        return output, avg_weights, entropy

class MeDeA(nn.Module):
    """MeDeA Pro: 集成 SE-ResNet Backbone 和 Multi-Query Heads"""
    def __init__(self, num_classes, d_model=256, base_filters=64, dropout=0.3, num_queries=4):
        super(MeDeA, self).__init__()       
        self.num_classes = num_classes
        
        # 🔥 升级：使用 Advanced SEResNet1d
        self.backbone = AdvancedSEResNet1d(
            input_channels=12, 
            base_filters=base_filters, 
            dropout=dropout,
            kernel_size=7 # 较大的卷积核有助于ECG波形特征提取
        )
        
        self.attention_heads = nn.ModuleList([
            MultiQueryAttentionHead(
                feature_dim=self.backbone.feature_dim, 
                d_model=d_model, 
                num_queries=num_queries,
                dropout=dropout
            )
            for _ in range(num_classes)
        ])
        
    def forward(self, x):
        feature_maps = self.backbone(x)
        
        outputs = []
        attention_weights = []
        total_entropy = 0.0
        
        for head in self.attention_heads:
            # head 返回: logit, avg_weights, entropy
            out, w, ent = head(feature_maps)
            outputs.append(out)
            attention_weights.append(w)
            total_entropy += ent
        
        final_output = torch.cat(outputs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        
        # 返回平均熵用于 Loss 计算 (Performance Boost)
        avg_entropy = total_entropy / self.num_classes
        
        return final_output, attention_weights, avg_entropy

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
            
            # MeDeA Pro 返回三个值，取第一个(logits)
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
            
            # MeDeA Pro 返回三个值，取第一个(logits)
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
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y_true_class, y_pred_class, zero_division=0)
        recall = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
        
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
    # 选择前3个导联进行显示
    signal_to_plot = signal[:3]  # (3, time_steps)
    signal_len = signal_to_plot.shape[1]
    
    fig = plt.figure(figsize=(20, 12))
    num_classes = len(classes)
    
    for class_idx in range(num_classes):
        ax = plt.subplot(num_classes, 1, class_idx + 1)
        
        class_name = classes[class_idx]
        prob = probabilities[class_idx]
        true_label = true_labels[class_idx]
        pred_label = pred_labels[class_idx]
        
        attention = attention_weights[class_idx]
        
        # 映射到信号长度
        attention_upsampled = np.interp(
            np.linspace(0, len(attention)-1, signal_len),
            np.arange(len(attention)),
            attention
        )
        
        attention_smooth = gaussian_filter1d(attention_upsampled, sigma=2.0)
        
        if attention_smooth.max() > attention_smooth.min():
            attention_normalized = (attention_smooth - attention_smooth.min()) / (attention_smooth.max() - attention_smooth.min())
        else:
            attention_normalized = np.zeros_like(attention_smooth)
        
        for lead_idx in range(signal_to_plot.shape[0]):
            offset = (lead_idx - 1) * 2
            signal_with_offset = signal_to_plot[lead_idx] + offset
            ax.plot(signal_with_offset, color='black', alpha=0.8, linewidth=1.0, 
                   label=f'Lead {lead_idx+1}' if lead_idx < 3 else None)
        
        x = np.arange(signal_len)
        y_min, y_max = ax.get_ylim()
        X, Y = np.meshgrid(x, np.linspace(y_min, y_max, 100))
        Z = np.tile(attention_normalized, (100, 1))
        
        alpha_intensity = min(0.7, prob * 1.5)
        cmap = plt.cm.Reds if true_label == 1 else plt.cm.Blues
            
        im = ax.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=alpha_intensity, 
                        vmin=0, vmax=1, extend='both')
        
        status_symbol = "✓" if (true_label == pred_label) else "✗"
        status_color = "green" if (true_label == pred_label) else "red"
        
        title = f'{class_name} | P={prob:.3f} | True={int(true_label)} | Pred={int(pred_label)} {status_symbol}'
        ax.set_title(title, fontsize=12, fontweight='bold', color=status_color if true_label != pred_label else 'black')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        if class_idx == 0:
            ax.legend(loc='upper right')
    
    plt.suptitle(f'Sample {sample_idx} - ECG Signal with Disease-Specific Attention Overlay', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 已保存注意力叠加图: {save_path}")

def create_combined_attention_overlay(signal, attention_weights, classes, probabilities, 
                                    true_labels, pred_labels, sample_idx, save_path):
    signal_to_plot = signal[:3]
    signal_len = signal_to_plot.shape[1]
    prob_threshold = 0.5
    
    combined_attention = np.zeros(signal_len)
    active_diseases = []
    
    print(f"🔍 样本 {sample_idx} 疾病概率分析:")
    for class_idx in range(len(classes)):
        class_name = classes[class_idx]
        prob = probabilities[class_idx]
        print(f"    - {class_name}: P={prob:.3f}")
        
        if prob > prob_threshold:
            attention = attention_weights[class_idx]
            attention_upsampled = np.interp(
                np.linspace(0, len(attention)-1, signal_len),
                np.arange(len(attention)),
                attention
            )
            attention_smooth = gaussian_filter1d(attention_upsampled, sigma=2.0)
            combined_attention += attention_smooth * prob
            active_diseases.append(f"{class_name}(P={prob:.2f})")
            print(f"      ✓ 已纳入综合注意力计算")
        else:
            print(f"      ✗ 概率过低，已排除")
    
    if combined_attention.max() > 0:
        combined_attention = combined_attention / combined_attention.max()
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    for lead_idx in range(signal_to_plot.shape[0]):
        offset = (lead_idx - 1) * 2
        signal_with_offset = signal_to_plot[lead_idx] + offset
        ax.plot(signal_with_offset, color='black', alpha=0.8, linewidth=1.5, label=f'Lead {lead_idx+1}')
    
    x = np.arange(signal_len)
    y_min, y_max = ax.get_ylim()
    X, Y = np.meshgrid(x, np.linspace(y_min, y_max, 100))
    Z = np.tile(combined_attention, (100, 1))
    
    im = ax.contourf(X, Y, Z, levels=50, cmap='plasma', alpha=0.6, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Combined Attention Weight (Active Diseases Only)', fontsize=12)
    
    active_diseases_str = ", ".join(active_diseases) if active_diseases else "No active diseases"
    title = f'Sample {sample_idx} - ECG Signal with Filtered Combined Attention\nActive Diseases: {active_diseases_str}'
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
    if not multi_label_samples:
        print("⚠️ 没有样本可用于生成类别比较图")
        return
    print(f"🎨 正在生成类别间注意力比较图...")
    
    fig = plt.figure(figsize=(20, 12))
    num_samples = min(3, len(multi_label_samples))
    
    for sample_idx in range(num_samples):
        sample = multi_label_samples[sample_idx]
        attention_weights = sample['attention_weights']
        probabilities = sample['probabilities']
        true_labels = sample['true_labels']
        signal_len = attention_weights.shape[1]
        
        ax = plt.subplot(num_samples, 1, sample_idx + 1)
        prob_threshold = 0.5
        active_indices = [i for i, prob in enumerate(probabilities) if prob > prob_threshold]
        if not active_indices:
            active_indices = np.argsort(probabilities)[-3:]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(active_indices)))
        for i, (class_idx, color) in enumerate(zip(active_indices, colors)):
            attention = attention_weights[class_idx]
            prob = probabilities[class_idx]
            true_label = true_labels[class_idx]
            class_name = classes[class_idx]
            attention_smooth = gaussian_filter1d(attention, sigma=2.0)
            
            linestyle = '-' if true_label == 1 else '--'
            linewidth = 3 if prob > 0.5 else 2
            alpha = 0.9 if prob > 0.5 else 0.6
            label = f'{class_name} (P={prob:.3f}, True={int(true_label)})'
            
            ax.plot(attention_smooth, label=label, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        
        ax.set_title(f'Sample {sample_idx + 1} - Active Disease Attention Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Attention Weight')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, signal_len)
    
    plt.suptitle('Disease-Specific Attention Patterns Comparison Across Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = explanations_dir / "class_attention_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 已保存类别间注意力比较图: {save_path}")

def generate_and_save_explanations(model, data_loader, classes, device, output_dir):
    print("🔍 开始生成多标签可解释性分析...")
    model.eval()
    explanations_dir = output_dir / "explanations"
    explanations_dir.mkdir(exist_ok=True)
    
    multi_label_samples = []
    with torch.no_grad():
        for signals, targets in data_loader:
            signals = signals.to(device, non_blocking=True)
            outputs, attention_weights, _ = model(signals) # 忽略 entropy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            for batch_idx in range(signals.size(0)):
                true_labels = targets[batch_idx].cpu().numpy()
                pred_labels = preds[batch_idx].cpu().numpy()
                prob_values = probs[batch_idx].cpu().numpy()
                
                if np.sum(true_labels) > 1 or np.sum(pred_labels) > 1:
                    multi_label_samples.append({
                        'signal': signals[batch_idx].cpu().numpy(),
                        'attention_weights': attention_weights[batch_idx].cpu().numpy(),
                        'true_labels': true_labels,
                        'pred_labels': pred_labels,
                        'probabilities': prob_values,
                        'num_true_labels': np.sum(true_labels),
                        'num_pred_labels': np.sum(pred_labels)
                    })
                if len(multi_label_samples) >= 10:
                    break
            if len(multi_label_samples) >= 10:
                break
    
    if not multi_label_samples:
        print("⚠️ 没有找到多标签样本，生成单标签示例...")
        with torch.no_grad():
            for signals, targets in data_loader:
                signals = signals.to(device, non_blocking=True)
                outputs, attention_weights, _ = model(signals)
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

    multi_label_samples.sort(key=lambda x: x['num_true_labels'], reverse=True)
    
    for sample_idx, sample in enumerate(multi_label_samples[:3]):
        print(f"🎨 正在生成样本 {sample_idx + 1} 的改进可解释性分析...")
        signal = sample['signal'][:3]
        attention_weights = sample['attention_weights']
        true_labels = sample['true_labels']
        pred_labels = sample['pred_labels']
        probabilities = sample['probabilities']
        signal_len = signal.shape[1]
        
        confirmed_diseases = [(i, classes[i]) for i, label in enumerate(true_labels) if label == 1]
        num_confirmed = len(confirmed_diseases)
        if num_confirmed == 0:
            top_prob_indices = np.argsort(probabilities)[-2:]
            confirmed_diseases = [(i, classes[i]) for i in top_prob_indices]
            num_confirmed = len(confirmed_diseases)
        
        total_rows = 1 + num_confirmed + 1
        fig = plt.figure(figsize=(24, total_rows * 3.5))
        gs = fig.add_gridspec(total_rows, 1, height_ratios=[1.2] + [0.8] * num_confirmed + [1.0], hspace=0.8)
        
        ax1 = fig.add_subplot(gs[0])
        for i in range(3):
            ax1.plot(signal[i], label=f'Lead {i+1}', alpha=0.8, linewidth=1.5)
        ax1.set_title(f'Sample {sample_idx + 1} - Original ECG Signal (First 3 Leads)', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Time Steps', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, signal_len)
        
        for heat_idx, (class_idx, class_name) in enumerate(confirmed_diseases):
            ax_heat = fig.add_subplot(gs[1 + heat_idx])
            attention = attention_weights[class_idx]
            prob = probabilities[class_idx]
            true_label = true_labels[class_idx]
            pred_label = pred_labels[class_idx]
            attention_upsampled = np.interp(np.linspace(0, len(attention)-1, signal_len), np.arange(len(attention)), attention)
            attention_smooth = gaussian_filter1d(attention_upsampled, sigma=1.5)
            attention_2d = attention_smooth.reshape(1, -1)
            
            im = ax_heat.imshow(attention_2d, cmap='viridis', aspect='auto', interpolation='bilinear', extent=[0, signal_len, 0.5, -0.5])
            
            status = "TP ✓" if true_label == 1 and pred_label == 1 else ("FN ✗" if true_label == 1 and pred_label == 0 else ("FP ✗" if true_label == 0 and pred_label == 1 else "TN ✓"))
            title_color = "green" if status.endswith("✓") else ("red" if "FN" in status else "orange")
            
            title = f'{class_name} Attention Heatmap | P={prob:.3f} | True={int(true_label)} | Pred={int(pred_label)} | {status}'
            ax_heat.set_title(title, fontsize=14, fontweight='bold', color=title_color, pad=15)
            ax_heat.set_xlabel('Time Steps', fontsize=12)
            ax_heat.set_ylabel('Attention', fontsize=12)
            ax_heat.set_yticks([0])
            ax_heat.set_yticklabels(['Weight'])
            ax_heat.set_xlim(0, signal_len)
            cbar = plt.colorbar(im, ax=ax_heat, orientation='horizontal', pad=0.4, shrink=0.6, aspect=30)
            cbar.set_label('Attention Weight', fontsize=10)
        
        ax_combined = fig.add_subplot(gs[1 + num_confirmed])
        combined_attention = np.zeros(signal_len)
        active_diseases_info = []
        for class_idx, class_name in confirmed_diseases:
            prob = probabilities[class_idx]
            attention = attention_weights[class_idx]
            attention_upsampled = np.interp(np.linspace(0, len(attention)-1, signal_len), np.arange(len(attention)), attention)
            combined_attention += gaussian_filter1d(attention_upsampled, sigma=2.0) * prob
            active_diseases_info.append(f"{class_name}(P={prob:.2f})")
        
        if combined_attention.max() > 0:
            combined_attention = combined_attention / combined_attention.max()
        
        combined_2d = combined_attention.reshape(1, -1)
        im_combined = ax_combined.imshow(combined_2d, cmap='viridis', aspect='auto', interpolation='bilinear', extent=[0, signal_len, 0.5, -0.5])
        combined_title = f'Probability-Weighted Combined Attention (Confirmed Diseases)\nActive: {", ".join(active_diseases_info)}'
        ax_combined.set_title(combined_title, fontsize=16, fontweight='bold', pad=20)
        ax_combined.set_xlabel('Time Steps', fontsize=14)
        ax_combined.set_ylabel('Combined\nAttention', fontsize=12)
        ax_combined.set_yticks([0])
        ax_combined.set_yticklabels(['Weighted'])
        ax_combined.set_xlim(0, signal_len)
        cbar_combined = plt.colorbar(im_combined, ax=ax_combined, orientation='horizontal', pad=0.3, shrink=0.6, aspect=30)
        cbar_combined.set_label('Combined Attention Weight', fontsize=12)
        
        confirmed_names = [name for _, name in confirmed_diseases]
        plt.suptitle(f'Sample {sample_idx + 1} - Disease-Specific Attention Analysis\nConfirmed Diseases: {", ".join(confirmed_names)} | Correctly Predicted: {np.sum(true_labels == pred_labels)}/{len(classes)}', fontsize=20, fontweight='bold', y=0.98)
        
        save_path = explanations_dir / f"enhanced_explanation_sample_{sample_idx + 1}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
        plt.close()
        print(f"✅ 已保存样本 {sample_idx + 1} 的改进可解释性分析图: {save_path}")
        
        print(f"🎨 正在生成样本 {sample_idx + 1} 的其他可解释性图...")
        create_attention_overlay_plot(signal=sample['signal'], attention_weights=sample['attention_weights'], classes=classes, probabilities=sample['probabilities'], true_labels=sample['true_labels'], pred_labels=sample['pred_labels'], sample_idx=sample_idx + 1, save_path=explanations_dir / f"attention_overlay_individual_sample_{sample_idx + 1}.png")
        create_combined_attention_overlay(signal=sample['signal'], attention_weights=sample['attention_weights'], classes=classes, probabilities=sample['probabilities'], true_labels=sample['true_labels'], pred_labels=sample['pred_labels'], sample_idx=sample_idx + 1, save_path=explanations_dir / f"attention_overlay_combined_sample_{sample_idx + 1}.png")
    
    print("🎨 正在生成类别间注意力比较图...")
    generate_class_comparison_plot(multi_label_samples[:5], classes, explanations_dir)
    print(f"🎉 改进的多标签可解释性分析完成！所有图像已保存到: {explanations_dir}")

def plot_confusion_matrices(y_true, y_pred, classes, output_dir):
    from sklearn.metrics import confusion_matrix
    output_dir = Path(output_dir)
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    print(f"🎨 正在生成混淆矩阵...")
    
    for i, class_name in enumerate(classes):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        cm = confusion_matrix(y_true_class, y_pred_class)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {class_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.savefig(cm_dir / f"confusion_matrix_{class_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for i, (class_name, ax) in enumerate(zip(classes, axes)):
        if i >= len(classes):
            ax.axis('off')
            continue
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        cm = confusion_matrix(y_true_class, y_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')
    for j in range(len(classes), len(axes)):
        axes[j].axis('off')
    plt.suptitle('Confusion Matrices for All Classes', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_dir / "confusion_matrices_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存到: {cm_dir}")

def plot_multilabel_confusion_matrix(y_true, y_pred, classes, output_dir):
    import pandas as pd
    output_dir = Path(output_dir)
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    print(f"🎨 正在生成多标签共现混淆矩阵...")
    
    n_classes = len(classes)
    prediction_rate_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        true_class_i_mask = y_true[:, i] == 1
        n_true_class_i = true_class_i_mask.sum()
        if n_true_class_i > 0:
            for j in range(n_classes):
                pred_class_j_in_true_i = (y_pred[true_class_i_mask, j] == 1).sum()
                prediction_rate_matrix[i, j] = pred_class_j_in_true_i / n_true_class_i
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(prediction_rate_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Prediction Rate'}, vmin=0, vmax=1)
    plt.title('Multi-Label Prediction Rate Matrix\n(Proportion of samples where Pred=1 | True=1)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Condition', fontsize=14)
    plt.ylabel('True Condition Present', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_dir / "multilabel_prediction_rate_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    cooccurrence_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_classes):
        for j in range(n_classes):
            cooccurrence_matrix[i, j] = ((y_true[:, i] == 1) & (y_pred[:, j] == 1)).sum()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence_matrix, annot=True, fmt='d', cmap='YlOrRd', xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Sample Count'})
    plt.title('Co-occurrence Count Matrix\n(Number of samples where True=1 AND Pred=1)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Condition', fontsize=14)
    plt.ylabel('True Condition Present', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_dir / "multilabel_cooccurrence_count_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(prediction_rate_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f'{prediction_rate_matrix[i, j]:.2f}', ha="center", va="center", color="white" if prediction_rate_matrix[i, j] > 0.5 else "black", fontsize=12, fontweight='bold')
    ax.set_title('Multi-Label Confusion Matrix\n(Prediction Rate: P(Pred=j | True=i))', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Condition Present', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prediction Rate', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_dir / "multilabel_confusion_matrix_styled.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 多标签混淆矩阵生成完成！")

# 在交叉验证中调用
def run_cross_validation(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 MeDeA-Pro (Advanced) 模型十折交叉验证...")
    print(f"    - 设备: {device}")
    print(f"    - 结果将保存至: {output_dir}")
    
    all_fold_results = {
        'val_f1': [], 'val_auc': [], 'test_f1': [], 'test_auc': [], 'fold_times': [], 'per_class_metrics': []
    }
    classes = None
    
    for fold in range(1, 11):
        print(f"\n{'='*60}\n🔄 开始第 {fold}/10 折交叉验证\n{'='*60}")
        fold_start_time = time.time()
        fold_output_dir = output_dir / f"fold_{fold}"
        fold_output_dir.mkdir(exist_ok=True)
        
        try:
            train_loader, val_loader, test_loader, fold_classes = create_dataloaders_cv(
                args.data_dir, fold, args.batch_size, args.num_workers
            )
            if classes is None: classes = fold_classes
        except FileNotFoundError as e:
            print(f"⚠️ 跳过第 {fold} 折: {e}")
            continue
        
        model = MeDeA(
            num_classes=len(classes),
            d_model=args.d_model,
            base_filters=args.base_filters,
            dropout=args.dropout,
            num_queries=args.num_queries
        ).to(device)
        
        if fold == 1:
            print(f"🏥 MeDeA-Pro 模型配置: Backbone=SE-ResNet1D, num_queries={args.num_queries}")
            print(f"   - Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        best_f1, patience_counter = 0.0, 0
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Fold {fold} Epoch {epoch+1}/{args.epochs}')
            
            for signals, targets in progress_bar:
                signals, targets = signals.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # 🔥 接收 avg_entropy
                outputs, attention_weights, avg_entropy = model(signals)
                
                classification_loss = criterion(outputs, targets)
                
                # 🔥 使用内部精确计算的 Entropy 进行正则化
                # -0.01 * entropy 意味着我们鼓励 entropy 更大 (更平滑的分布) 以避免过拟合
                # 如果你发现模型过于“平滑”，可以减小系数或者变为正号
                attention_entropy_loss = -0.01 * avg_entropy 
                
                total_loss_batch = classification_loss + attention_entropy_loss
                
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                progress_bar.set_postfix({
                    'cls_loss': f'{classification_loss.item():.4f}',
                    'ent_loss': f'{attention_entropy_loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

            val_f1, val_auc, _, _, _ = run_evaluation_detailed(model, val_loader, device, classes)
            scheduler.step(val_f1)
            
            if args.verbose or epoch % 10 == 0:
                print(f"Fold {fold} Epoch {epoch+1}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Loss: {total_loss / len(train_loader):.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), fold_output_dir / 'best_medea_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"⌛️ Fold {fold} 早停: 在第 {epoch+1} 个epoch触发")
                break
        
        model.load_state_dict(torch.load(fold_output_dir / 'best_medea_model.pth', map_location=device))
        test_f1, test_auc, test_true, test_preds, per_class_metrics = run_evaluation_detailed(model, test_loader, device, classes)
        
        np.savez(fold_output_dir / 'predictions.npz', y_true=test_true, y_pred=test_preds, classes=classes)
        plot_confusion_matrices(test_true, test_preds, classes, fold_output_dir)
        plot_multilabel_confusion_matrix(test_true, test_preds, classes, fold_output_dir)
        
        fold_time = time.time() - fold_start_time
        all_fold_results['val_f1'].append(best_f1)
        all_fold_results['val_auc'].append(val_auc)
        all_fold_results['test_f1'].append(test_f1)
        all_fold_results['test_auc'].append(test_auc)
        all_fold_results['fold_times'].append(fold_time)
        all_fold_results['per_class_metrics'].append(per_class_metrics)
        
        print(f"\n📊 第 {fold} 折结果: Val F1: {best_f1:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Time: {fold_time/60:.2f} min")
        
        with open(fold_output_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Fold {fold} Classification Report\n" + "="*50 + "\n")
            f.write(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
            f.write(f"\nTest F1-Score: {test_f1:.4f}\nTest AUC: {test_auc:.4f}\n")
        
        if args.run_explain and fold in args.explain_folds:
            generate_and_save_explanations(model=model, data_loader=test_loader, classes=classes, device=device, output_dir=fold_output_dir)
    
    if all_fold_results['test_f1']:
        print(f"\n{'='*60}\n🎉 十折交叉验证完成！\n{'='*60}")
        mean_test_f1 = np.mean(all_fold_results['test_f1'])
        std_test_f1 = np.std(all_fold_results['test_f1'])
        print(f"📈 总体测试 F1-Score: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
        
        # 保存汇总结果 (略去繁琐的打印，与之前一致)
        import json
        with open(output_dir / 'cross_validation_summary.json', 'w') as f:
            json.dump({'mean_test_f1': mean_test_f1, 'std_test_f1': std_test_f1, 'details': all_fold_results}, f, indent=2, default=str)

def main_training_single_fold(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 MeDeA-Pro 模型训练 (Fold {args.fold})...")
    train_loader, val_loader, test_loader, classes = create_dataloaders_cv(args.data_dir, args.fold, args.batch_size, args.num_workers)
    
    model = MeDeA(
        num_classes=len(classes),
        d_model=args.d_model,
        base_filters=args.base_filters,
        dropout=args.dropout,
        num_queries=args.num_queries
    ).to(device)
    
    print(f"🏥 MeDeA-Pro 配置: Backbone=SE-ResNet1D, Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
            
            outputs, attention_weights, avg_entropy = model(signals) # 🔥 接收 entropy
            
            classification_loss = criterion(outputs, targets)
            attention_entropy_loss = -0.01 * avg_entropy # 🔥 使用内部 entropy
            
            total_loss_batch = classification_loss + attention_entropy_loss
            
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            progress_bar.set_postfix({'cls': f'{classification_loss.item():.4f}', 'ent': f'{avg_entropy.item():.2f}'})

        val_f1, _, _, _ = run_evaluation(model, val_loader, device)
        scheduler.step(val_f1)
        print(f"Epoch {epoch+1}, Val F1: {val_f1:.4f}, Loss: {total_loss / len(train_loader):.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_medea_model.pth')
            print(f"📈 新最佳 F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"⌛️ 早停")
            break
            
    print(f"\n🚀 测试集评估 (Fold {args.fold})...")
    model.load_state_dict(torch.load(output_dir / 'best_medea_model.pth', map_location=device))
    test_f1, test_auc, test_true, test_preds = run_evaluation(model, test_loader, device)

    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
    print(f"Macro F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    np.savez(output_dir / 'predictions.npz', y_true=test_true, y_pred=test_preds, classes=classes)
    plot_confusion_matrices(test_true, test_preds, classes, output_dir)
    plot_multilabel_confusion_matrix(test_true, test_preds, classes, output_dir)

    if args.run_explain:
        generate_and_save_explanations(model=model, data_loader=test_loader, classes=classes, device=device, output_dir=output_dir)

def run_explain_only(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    model_path = output_dir / 'best_medea_model.pth'
    
    if not model_path.exists(): raise FileNotFoundError(f"❌ 找不到模型: {model_path}")
    print(f"🔍 启动解释模式 (Fold {args.fold})...")
    
    _, _, test_loader, classes = create_dataloaders_cv(args.data_dir, args.fold, args.batch_size, args.num_workers)
    model = MeDeA(num_classes=len(classes), d_model=args.d_model, base_filters=args.base_filters, dropout=args.dropout, num_queries=args.num_queries).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    generate_and_save_explanations(model=model, data_loader=test_loader, classes=classes, device=device, output_dir=output_dir)

# ==============================================================================
# 7. 主程序入口 (Main Entry)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🚀 MeDeA-Pro (SE-ResNet + MultiQuery) 训练脚本')
    
    parser.add_argument('--data_dir', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data')
    parser.add_argument('--output_dir', type=str, default='./saved_models/medea_advanced_se')
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--explain_folds', type=int, nargs='+', default=[1])
    parser.add_argument('--verbose', action='store_true')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_queries', type=int, default=8, help='每个头的查询数量 (SOTA建议设为8或16)')

    parser.add_argument('--run_explain', action='store_true')
    parser.add_argument('--explain_only', action='store_true')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if args.explain_only:
        args.fold = args.fold if hasattr(args, 'fold') else 1
        run_explain_only(args)
    elif args.cross_validation:
        run_cross_validation(args)
    else:
        main_training_single_fold(args)