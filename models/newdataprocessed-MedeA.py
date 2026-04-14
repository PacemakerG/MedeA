# -*- coding: utf-8 -*-
"""
使用 MeDeA (Multi-disease Decompositional Attention) 模型训练 Chapman 数据集
- 单导联 Lead II
- 5分类任务（SR, SB, ST, AF+AFib, AT+SVT）
- 500 Hz 采样率
- 10秒信号（5000个采样点）
"""
import os
import argparse
import random
import time
from datetime import datetime
from pathlib import Path
import math
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
class ChapmanDataset(Dataset):
    """为Chapman ECG数据创建PyTorch Dataset。"""
    def __init__(self, X, y):
        # X shape: (N, 1, 5000) - 单导联, 5000采样点
        # y shape: (N, 5) - one-hot编码的5分类标签
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders_cv(data_dir, fold_num, batch_size, num_workers):
    """从五折交叉验证数据文件夹加载指定fold的数据并创建DataLoaders。"""
    data_dir = Path(data_dir)
    
    # 检查目录是否存在
    if not data_dir.exists():
        raise FileNotFoundError(f"❌ 数据目录未找到: {data_dir}")
    
    if not data_dir.is_dir():
        raise NotADirectoryError(f"❌ 路径不是目录: {data_dir}")
    
    # 构建完整的文件路径
    data_file_path = data_dir / f"fold{fold_num}.npz"
    
    # 检查文件是否存在
    if not data_file_path.exists():
        raise FileNotFoundError(f"❌ 数据文件未找到: {data_file_path}")
    
    print(f"⌛️ 正在从 {data_file_path} 加载第 {fold_num} 折数据...")
    data = np.load(data_file_path, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    
    print(f"📊 数据形状检查:")
    print(f"   - X_train: {X_train.shape} (期望: (N, 1, 5000))")
    print(f"   - y_train: {y_train.shape} (期望: (N, 5))")
    
    train_ds = ChapmanDataset(X_train, y_train)
    val_ds = ChapmanDataset(X_val, y_val)
    test_ds = ChapmanDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
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
    """共享的1D ResNet骨干网络 - 适配单导联输入"""
    def __init__(self, input_channels=1, base_filters=64, dropout=0.1):  # 修改为1导联
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
        # x shape: (batch_size, 1, 5000) - 单导联, 5000采样点
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
    """注意力头 - 为每个疾病类别单独计算注意力"""
    def __init__(self, feature_dim, d_model, dropout=0.1):
        super(AttentionHead, self).__init__()
        self.d_model = d_model
        
        # 注意力机制
        self.query = nn.Linear(feature_dim, d_model)
        self.key = nn.Linear(feature_dim, d_model)
        self.value = nn.Linear(feature_dim, d_model)
        
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
        attention_weights = F.softmax(attention_scores, dim=-1)  
        
        # 应用注意力
        attended_features = torch.matmul(attention_weights, V)  # (batch_size, 1, d_model)
        attended_features = attended_features.squeeze(1)  # (batch_size, d_model)
        
        # 残差连接和层标准化
        attended_features = self.layer_norm(attended_features + self.query(pooled_features))
        attended_features = self.dropout(attended_features)
        
        # 分类
        output = self.classifier(attended_features)
        
        return output  # output: (batch_size, 1)

class MeDeA(nn.Module):
    """Multi-disease Decompositional Attention 模型 - 适配Chapman五分类任务"""
    def __init__(self, num_classes=5, d_model=256, base_filters=64, dropout=0.3):
        super(MeDeA, self).__init__()      
        self.num_classes = num_classes
        
        # 共享骨干网络 - 修改为单导联输入
        self.backbone = SharedBackbone(input_channels=1, base_filters=base_filters, dropout=dropout)
        
        # 为每个疾病创建专门的注意力头
        self.attention_heads = nn.ModuleList([
            AttentionHead(self.backbone.feature_dim, d_model, dropout)
            for _ in range(num_classes)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, 1, 5000) - 单导联, 5000采样点
        pooled_features, feature_maps = self.backbone(x)
        
        outputs = []
        
        # 为每个疾病类别计算输出
        for attention_head in self.attention_heads:
            output = attention_head(pooled_features, feature_maps)
            outputs.append(output)
        
        # 合并输出
        final_output = torch.cat(outputs, dim=1)  # (batch_size, num_classes)
        
        return final_output

# ==============================================================================
# 4. 训练与评估逻辑 (Training & Evaluation)
# ==============================================================================
def run_evaluation(model, data_loader, device):
    """在验证集或测试集上运行评估。"""
    model.eval()
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for signals, targets in data_loader:
            signals = signals.to(device, non_blocking=True)
            outputs = model(signals)
            
            # 对于多分类,使用softmax + argmax
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # 将one-hot转为类别索引
            targets_idx = torch.argmax(targets, dim=1)
            
            all_targets.append(targets_idx.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    
    # 计算多分类指标
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return accuracy, macro_f1, macro_precision, macro_recall, all_targets, all_preds

def run_evaluation_detailed(model, data_loader, device, classes):
    """在验证集或测试集上运行详细评估,返回每个类别的性能指标。"""
    model.eval()
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for signals, targets in data_loader:
            signals = signals.to(device, non_blocking=True)
            outputs = model(signals)
            
            # 对于多分类,使用softmax + argmax
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # 将one-hot转为类别索引
            targets_idx = torch.argmax(targets, dim=1)
            
            all_targets.append(targets_idx.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    
    # 计算每个类别的性能指标
    per_class_metrics = {}
    
    for i, class_name in enumerate(classes):
        # 为每个类别创建二分类问题
        y_true_binary = (all_targets == i).astype(int)
        y_pred_binary = (all_preds == i).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    # 计算宏平均
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return accuracy, macro_f1, macro_precision, macro_recall, all_targets, all_preds, per_class_metrics

# ==============================================================================
# 5. 交叉验证主函数
# ==============================================================================
def run_cross_validation(args):
    """运行五折交叉验证。"""
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 MeDeA 模型五折交叉验证 (Chapman 五分类)...")
    print(f"   - 设备: {device}")
    print(f"   - 结果将保存至: {output_dir}")
    
    # 存储所有fold的结果
    all_fold_results = {
        'val_accuracy': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'test_accuracy': [],
        'test_f1': [],
        'test_precision': [],
        'test_recall': [],
        'fold_times': [],
        'per_class_metrics': []
    }
    
    classes = None
    
    # 运行五折交叉验证
    for fold in range(1, args.n_folds + 1):
        print(f"\n{'='*60}")
        print(f"🔄 开始第 {fold}/{args.n_folds} 折交叉验证")
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
                classes = fold_classes
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
        
        if fold == 1:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"🏥 MeDeA 模型配置: d_model={args.d_model}, base_filters={args.base_filters}")
            print(f"   - 可训练参数: {param_count:,}")

        criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        
        best_f1, patience_counter = 0.0, 0
        
        # 训练循环
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Fold {fold} Epoch {epoch+1}/{args.epochs}')
            
            for signals, targets in progress_bar:
                signals = signals.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # 将one-hot转为类别索引
                targets_idx = torch.argmax(targets, dim=1)
                
                outputs = model(signals)
                loss = criterion(outputs, targets_idx)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

            val_acc, val_f1, val_prec, val_rec, _, _ = run_evaluation(model, val_loader, device)
            scheduler.step(val_f1)
            
            if args.verbose or epoch % 10 == 0:
                print(f"Fold {fold} Epoch {epoch+1}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Loss: {total_loss / len(train_loader):.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), fold_output_dir / 'best_medea_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"⌛️ Fold {fold} 早停: 在第 {epoch+1} 个epoch触发")
                break
        
        # 在测试集上评估
        model.load_state_dict(torch.load(fold_output_dir / 'best_medea_model.pth', map_location=device))
        test_acc, test_f1, test_prec, test_rec, test_true, test_preds, per_class_metrics = run_evaluation_detailed(
            model, test_loader, device, classes
        )
        
        # 保存预测结果
        np.savez(
            fold_output_dir / 'predictions.npz',
            y_true=test_true,
            y_pred=test_preds,
            classes=classes
        )
        
        fold_time = time.time() - fold_start_time
        
        # 保存当前fold的结果
        all_fold_results['val_accuracy'].append(val_acc)
        all_fold_results['val_f1'].append(best_f1)
        all_fold_results['val_precision'].append(val_prec)
        all_fold_results['val_recall'].append(val_rec)
        all_fold_results['test_accuracy'].append(test_acc)
        all_fold_results['test_f1'].append(test_f1)
        all_fold_results['test_precision'].append(test_prec)
        all_fold_results['test_recall'].append(test_rec)
        all_fold_results['fold_times'].append(fold_time)
        all_fold_results['per_class_metrics'].append(per_class_metrics)
        
        print(f"\n📊 第 {fold} 折结果:")
        print(f"   - 测试准确率: {test_acc:.4f}")
        print(f"   - 测试 F1: {test_f1:.4f}")
        print(f"   - 测试精确率: {test_prec:.4f}")
        print(f"   - 测试召回率: {test_rec:.4f}")
        print(f"   - 用时: {fold_time/60:.2f} 分钟")
        
        # 保存详细的分类报告
        with open(fold_output_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Fold {fold} Classification Report\n")
            f.write("="*50 + "\n")
            f.write(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
            f.write(f"\nTest Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1-Score: {test_f1:.4f}\n")
            f.write(f"Test Precision: {test_prec:.4f}\n")
            f.write(f"Test Recall: {test_rec:.4f}\n")
            
            f.write(f"\nPer-Class Performance:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-"*50 + "\n")
            for class_name, metrics in per_class_metrics.items():
                f.write(f"{class_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1_score']:<10.4f}\n")
    
    # 计算并保存交叉验证总结果
    if all_fold_results['test_f1'] and classes is not None:
        print(f"\n{'='*60}")
        print("🎉 五折交叉验证完成!")
        print(f"{'='*60}")
        
        # 计算统计
        mean_test_acc = np.mean(all_fold_results['test_accuracy'])
        std_test_acc = np.std(all_fold_results['test_accuracy'])
        mean_test_f1 = np.mean(all_fold_results['test_f1'])
        std_test_f1 = np.std(all_fold_results['test_f1'])
        mean_test_prec = np.mean(all_fold_results['test_precision'])
        std_test_prec = np.std(all_fold_results['test_precision'])
        mean_test_rec = np.mean(all_fold_results['test_recall'])
        std_test_rec = np.std(all_fold_results['test_recall'])
        total_time = sum(all_fold_results['fold_times'])
        
        # 计算每个类别的平均性能
        per_class_summary = {}
        for class_name in classes:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for fold_metrics in all_fold_results['per_class_metrics']:
                if class_name in fold_metrics:
                    precision_scores.append(fold_metrics[class_name]['precision'])
                    recall_scores.append(fold_metrics[class_name]['recall'])
                    f1_scores.append(fold_metrics[class_name]['f1_score'])
            
            per_class_summary[class_name] = {
                'precision_mean': np.mean(precision_scores),
                'precision_std': np.std(precision_scores),
                'recall_mean': np.mean(recall_scores),
                'recall_std': np.std(recall_scores),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores)
            }
        
        print(f"📈 交叉验证结果总结:")
        print(f"   - 测试准确率: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"   - 测试 F1-Score: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"   - 测试精确率: {mean_test_prec:.4f} ± {std_test_prec:.4f}")
        print(f"   - 测试召回率: {mean_test_rec:.4f} ± {std_test_rec:.4f}")
        print(f"   - 总用时: {total_time/60:.2f} 分钟")
        
        print(f"\n📊 Per-Class Performance Summary:")
        print(f"{'Class':<20} {'Precision':<20} {'Recall':<20} {'F1-Score':<20}")
        print("-" * 80)
        for class_name, metrics in per_class_summary.items():
            precision_str = f"{metrics['precision_mean']:.3f}±{metrics['precision_std']:.3f}"
            recall_str = f"{metrics['recall_mean']:.3f}±{metrics['recall_std']:.3f}"
            f1_str = f"{metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f}"
            
            print(f"{class_name:<20} {precision_str:<20} {recall_str:<20} {f1_str:<20}")
        
        # 保存结果
        results_summary = {
            'cross_validation_results': all_fold_results,
            'summary_statistics': {
                'mean_test_accuracy': mean_test_acc,
                'std_test_accuracy': std_test_acc,
                'mean_test_f1': mean_test_f1,
                'std_test_f1': std_test_f1,
                'mean_test_precision': mean_test_prec,
                'std_test_precision': std_test_prec,
                'mean_test_recall': mean_test_rec,
                'std_test_recall': std_test_rec,
                'total_time_minutes': total_time/60
            },
            'per_class_summary': per_class_summary
        }
        
        with open(output_dir / 'cross_validation_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 保存详细文本报告
        with open(output_dir / 'cross_validation_report.txt', 'w') as f:
            f.write("MeDeA Model - Chapman 5-Class Classification Results\n")
            f.write("="*80 + "\n\n")
            
            f.write("Overall Performance Summary:\n")
            f.write("-"*40 + "\n")
            f.write(f"Test Accuracy:  {mean_test_acc:.4f} ± {std_test_acc:.4f}\n")
            f.write(f"Test F1-Score:  {mean_test_f1:.4f} ± {std_test_f1:.4f}\n")
            f.write(f"Test Precision: {mean_test_prec:.4f} ± {std_test_prec:.4f}\n")
            f.write(f"Test Recall:    {mean_test_rec:.4f} ± {std_test_rec:.4f}\n")
            f.write(f"Total Time:     {total_time/60:.2f} minutes\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("="*80 + "\n")
            f.write(f"{'Class':<20} {'Precision':<20} {'Recall':<20} {'F1-Score':<20}\n")
            f.write("-"*80 + "\n")
            
            for class_name, metrics in per_class_summary.items():
                precision_str = f"{metrics['precision_mean']*100:.2f}%"
                recall_str = f"{metrics['recall_mean']*100:.2f}%"
                f1_str = f"{metrics['f1_mean']*100:.2f}%"
                
                f.write(f"{class_name:<20} {precision_str:<20} {recall_str:<20} {f1_str:<20}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            
            f.write("Individual Fold Results:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Fold':<6} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Time(min)':<10}\n")
            f.write("-"*80 + "\n")
            for i, (acc, f1, prec, rec, t) in enumerate(zip(
                all_fold_results['test_accuracy'], all_fold_results['test_f1'],
                all_fold_results['test_precision'], all_fold_results['test_recall'],
                all_fold_results['fold_times']
            ), 1):
                f.write(f"Fold {i:<2} {acc:<10.4f} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f} {t/60:<10.1f}\n")
        
        print(f"📁 所有结果已保存到: {output_dir}")
    else:
        print("❌ 没有成功完成任何fold的训练")

# ==============================================================================
# 6. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🚀 MeDeA 模型训练脚本 (Chapman 五分类)')
    
    parser.add_argument('--data_dir', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/models/processeddata_5fold', 
                       help='包含五折交叉验证数据的目录路径')
    parser.add_argument('--output_dir', type=str, default='./saved_models/medea_chapman', 
                       help='保存模型和结果的目录')
    
    parser.add_argument('--n_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--fold', type=int, default=1, help='指定训练单个fold (1-5)')
    parser.add_argument('--cross_validation', action='store_true', help='运行五折交叉验证')
    parser.add_argument('--verbose', action='store_true', help='显示详细的训练过程')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--d_model', type=int, default=256, help='注意力头的隐藏维度')
    parser.add_argument('--base_filters', type=int, default=64, help='CNN骨干网络的基础滤波器数量')
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    
    args = parser.parse_args()
    
    if args.cross_validation:
        run_cross_validation(args)
    else:
        print("请使用 --cross_validation 参数运行五折交叉验证")
        print("示例: python MedeA.py --cross_validation --data_dir ./exgnet_chapman_500hz")