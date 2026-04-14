# -*- coding: utf-8 -*-
"""
使用 MeDeA (Single-Query) 模型训练 PTB-XL 数据集

Single-Query版本：使用单个查询向量替代Multi-Query的多个疾病特异性查询向量
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# ==============================================================================
# 1. 可复现性设置 (Reproducibility)
# ==============================================================================
def seed_everything(seed: int):
    """设置随机种子以确保可复现性。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 2. 数据加载模块 (Data Loading)
# ==============================================================================
class PTBXLDataset(Dataset):
    """PTB-XL 数据集类"""
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def create_dataloaders_cv(data_dir, fold_num, batch_size, num_workers):
    """从十折交叉验证数据文件夹加载指定fold的数据并创建DataLoaders。"""
    data_file_path = Path(data_dir) / f"ptbxl_processed_100hz_fold{fold_num}.npz"
    
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
# 3. MeDeA Single-Query 模型架构
# ==============================================================================

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器"""
    def __init__(self, input_channels=12, base_filters=64, dropout=0.3):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            # Block 1
            nn.Sequential(
                nn.Conv1d(input_channels, base_filters, kernel_size=7, padding=3),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout)
            ),
            # Block 2
            nn.Sequential(
                nn.Conv1d(base_filters, base_filters*2, kernel_size=5, padding=2),
                nn.BatchNorm1d(base_filters*2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout)
            ),
            # Block 3
            nn.Sequential(
                nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_filters*4),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout)
            )
        ])
        
        self.output_dim = base_filters * 4
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x.transpose(1, 2)  # (B, T, C)

class SingleQueryAttention(nn.Module):
    """🔥 Single-Query 注意力模块 - 使用单个查询向量"""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # 🔥 Single Query: 只有一个查询向量，不是每个类别一个
        self.single_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Key和Value的投影层
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 用于将单一注意力输出映射到多个类别的分类层
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = d_model ** -0.5
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # 🔥 扩展单个查询到batch size
        query = self.single_query.expand(batch_size, -1, -1)  # (B, 1, d_model)
        
        # 计算keys和values
        keys = self.key_proj(x)      # (B, T, d_model)
        values = self.value_proj(x)  # (B, T, d_model)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale  # (B, 1, T)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, 1, T)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, values)  # (B, 1, d_model)
        
        # 输出投影
        output = self.output_proj(attended_values)  # (B, 1, d_model)
        output = output.squeeze(1)  # (B, d_model)
        
        # 🔥 通过分类层映射到多个类别
        logits = self.classifier(output)  # (B, num_classes)
        
        # 🔥 返回注意力权重 - 需要扩展到每个类别以保持与Multi-Query版本的接口一致
        # 对于可解释性分析，我们复制同一个注意力权重给所有类别
        expanded_attention = attention_weights.squeeze(1).unsqueeze(1).expand(-1, self.num_classes, -1)  # (B, num_classes, T)
        
        return logits, expanded_attention

class MeDeASingleQuery(nn.Module):
    """🔥 MeDeA Single-Query 模型"""
    def __init__(self, num_classes=5, d_model=256, base_filters=64, dropout=0.3):
        super().__init__()
        
        # CNN特征提取器
        self.feature_extractor = CNNFeatureExtractor(
            input_channels=12, 
            base_filters=base_filters, 
            dropout=dropout
        )
        
        # 特征维度映射
        self.feature_proj = nn.Linear(self.feature_extractor.output_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 🔥 Single-Query 注意力模块
        self.attention = SingleQueryAttention(d_model, num_classes)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # CNN特征提取
        features = self.feature_extractor(x)  # (B, T, feature_dim)
        
        # 特征投影
        features = self.feature_proj(features)  # (B, T, d_model)
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        # 位置编码
        features = self.pos_encoding(features)
        
        # 🔥 Single-Query 注意力
        logits, attention_weights = self.attention(features)
        
        return logits, attention_weights

# ==============================================================================
# 4. 训练和评估函数
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
    f1_macro = f1_score(all_targets, preds, average='macro', zero_division=0)
    
    try:
        auc_macro = roc_auc_score(all_targets, all_probs, average='macro')
    except ValueError:
        auc_macro = 0.0
        
    return f1_macro, auc_macro, all_targets, preds

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
# 5. 交叉验证训练函数
# ==============================================================================

def run_cross_validation(args):
    """运行十折交叉验证。"""
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 MeDeA Single-Query 模型十折交叉验证...")
    print(f"   - 设备: {device}")
    print(f"   - 结果将保存至: {output_dir}")
    
    # 存储所有fold的结果
    all_fold_results = {
        'val_f1': [],
        'val_auc': [],
        'test_f1': [],
        'test_auc': [],
        'fold_times': [],
        'per_class_metrics': []
    }
    
    classes = None
    
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
                classes = fold_classes
        except FileNotFoundError as e:
            print(f"⚠️ 跳过第 {fold} 折: {e}")
            continue
        
        # 🔥 创建 Single-Query 模型
        model = MeDeASingleQuery(
            num_classes=len(classes),
            d_model=args.d_model,
            base_filters=args.base_filters,
            dropout=args.dropout
        ).to(device)
        
        if fold == 1:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"🏥 MeDeA Single-Query 模型配置: d_model={args.d_model}, base_filters={args.base_filters}")
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
                
                # 🔥 Single-Query版本的注意力正则化 - 不需要分散性约束
                attention_entropy = -torch.sum(
                    attention_weights.mean(dim=1) * torch.log(attention_weights.mean(dim=1) + 1e-8), 
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
            
            if args.verbose or epoch % 10 == 0:
                print(f"Fold {fold} Epoch {epoch+1}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Loss: {total_loss / len(train_loader):.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), fold_output_dir / 'best_medea_single_query_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"⌛️ Fold {fold} 早停: 在第 {epoch+1} 个epoch触发")
                break
        
        # 在测试集上评估最终模型
        model.load_state_dict(torch.load(fold_output_dir / 'best_medea_single_query_model.pth', map_location=device))
        test_f1, test_auc, test_true, test_preds, per_class_metrics = run_evaluation_detailed(model, test_loader, device, classes)
        
        fold_time = time.time() - fold_start_time
        
        # 保存当前fold的结果
        all_fold_results['val_f1'].append(best_f1)
        all_fold_results['val_auc'].append(val_auc)
        all_fold_results['test_f1'].append(test_f1)
        all_fold_results['test_auc'].append(test_auc)
        all_fold_results['fold_times'].append(fold_time)
        all_fold_results['per_class_metrics'].append(per_class_metrics)
        
        print(f"\n📊 第 {fold} 折结果:")
        print(f"   - 最佳验证 F1: {best_f1:.4f}")
        print(f"   - 测试 F1: {test_f1:.4f}")
        print(f"   - 测试 AUC: {test_auc:.4f}")
        print(f"   - 用时: {fold_time/60:.2f} 分钟")
        
        # 保存详细的分类报告
        with open(fold_output_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Fold {fold} Classification Report (Single-Query)\n")
            f.write("="*50 + "\n")
            f.write(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
            f.write(f"\nTest F1-Score: {test_f1:.4f}\n")
            f.write(f"Test AUC: {test_auc:.4f}\n")
            
            # 添加每个类别的详细指标
            f.write(f"\nPer-Class Performance:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}\n")
            f.write("-"*50 + "\n")
            for class_name, metrics in per_class_metrics.items():
                f.write(f"{class_name:<8} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1_score']:<10.4f} {metrics['auc']:<10.4f}\n")
    
    # 计算并保存详细的交叉验证总结果
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
        
        # 计算每个类别的平均性能指标
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
        
        print(f"📈 交叉验证结果总结 (Single-Query):")
        print(f"   - 验证 F1-Score: {mean_val_f1:.4f} ± {std_val_f1:.4f}")
        print(f"   - 测试 F1-Score: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"   - 测试 AUC:      {mean_test_auc:.4f} ± {std_test_auc:.4f}")
        print(f"   - 总用时: {total_time/60:.2f} 分钟")
        
        # 🔥 修复打印每个类别的平均性能的格式化问题
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
            'model_type': 'MeDeA_Single_Query',
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
            'per_class_summary': per_class_summary
        }
        
        # 保存为JSON文件
        with open(output_dir / 'cross_validation_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 保存详细的文本报告
        with open(output_dir / 'cross_validation_report.txt', 'w') as f:
            f.write("MeDeA Single-Query Model - 10-Fold Cross Validation Results\n")
            f.write("="*80 + "\n\n")
            
            # 总体性能统计
            f.write("Overall Performance Summary:\n")
            f.write("-"*40 + "\n")
            f.write(f"Validation F1-Score: {mean_val_f1:.4f} ± {std_val_f1:.4f}\n")
            f.write(f"Test F1-Score:       {mean_test_f1:.4f} ± {std_test_f1:.4f}\n")
            f.write(f"Test AUC:            {mean_test_auc:.4f} ± {std_test_auc:.4f}\n")
            f.write(f"Total Time:          {total_time/60:.2f} minutes\n\n")
            
            # Per-Class Performance Table
            f.write("TABLE II\n")
            f.write("MAIN PERFORMANCE COMPARISON ON THE PTB-XL TEST SET.\n")
            f.write("MeDeA (Single-Query) Results:\n")
            f.write("="*80 + "\n")
            f.write(f"{'Class':<8} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'AUC':<15}\n")
            f.write("-"*80 + "\n")
            
            for class_name, metrics in per_class_summary.items():
                precision_str = f"{metrics['precision_mean']*100:.2f}%"
                recall_str = f"{metrics['recall_mean']*100:.2f}%"
                f1_str = f"{metrics['f1_mean']*100:.2f}%"
                auc_str = f"{metrics['auc_mean']*100:.2f}%"
                
                f.write(f"{class_name:<8} {precision_str:<15} {recall_str:<15} {f1_str:<15} {auc_str:<15}\n")
            
            # 总体宏平均
            f.write("-"*80 + "\n")
            f.write(f"{'Macro F1':<8} {'':<15} {'':<15} {mean_test_f1*100:.2f}%{'':<10} {'':<15}\n")
            f.write(f"{'Macro AUC':<8} {'':<15} {'':<15} {'':<15} {mean_test_auc*100:.2f}%{'':<10}\n")
            
            # 参数数量计算 (需要重新创建模型来计算)
            temp_model = MeDeASingleQuery(
                num_classes=len(classes),
                d_model=args.d_model,
                base_filters=args.base_filters,
                dropout=args.dropout
            )
            param_count = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
            f.write(f"{'Params (M)':<8} {'':<15} {'':<15} {'':<15} {param_count/1e6:.1f}M{'':<10}\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        print(f"📁 所有结果已保存到: {output_dir}")
        print(f"📊 Single-Query模型性能报告已保存到: {output_dir}/cross_validation_report.txt")
    else:
        print("❌ 没有成功完成任何fold的训练")

def main_training_single_fold(args):
    """训练单个指定的fold。"""
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 MeDeA Single-Query 模型训练 (Fold {args.fold})...")
    print(f"   - 设备: {device}")
    print(f"   - 模型将保存至: {output_dir}")
    
    train_loader, val_loader, test_loader, classes = create_dataloaders_cv(
        args.data_dir, args.fold, args.batch_size, args.num_workers
    )
    
    model = MeDeASingleQuery(
        num_classes=len(classes),
        d_model=args.d_model,
        base_filters=args.base_filters,
        dropout=args.dropout
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏥 MeDeA Single-Query 模型配置: d_model={args.d_model}, base_filters={args.base_filters}")
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
                attention_weights.mean(dim=1) * torch.log(attention_weights.mean(dim=1) + 1e-8), 
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
            torch.save(model.state_dict(), output_dir / 'best_medea_single_query_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"⌛️ 早停: 在 {epoch+1} 个epoch触发")
            break
            
    total_time = time.time() - start_time
    print(f"\n✅ 训练完成，总耗时: {total_time/60:.2f} 分钟。")

    print(f"\n🚀 在测试集上评估最终模型 (Fold {args.fold})...")
    model.load_state_dict(torch.load(output_dir / 'best_medea_single_query_model.pth', map_location=device))
    test_f1, test_auc, test_true, test_preds = run_evaluation(model, test_loader, device)

    print(f"\n--- 🚀 MeDeA Single-Query 模型最终分类报告 (Fold {args.fold}) ---")
    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
    print(f"\n--- 🚀 Test Set Performance ---")
    print(f"Macro F1-Score: {test_f1:.4f}")
    print(f"Macro AUC:      {test_auc:.4f}")

# ==============================================================================
# 7. 可解释性分析模块 (Single-Query版本) - 移动到主程序之前
# ==============================================================================

def calculate_gini_coefficient(attention_weights):
    """
    计算基尼系数来衡量注意力分布的不平等性
    基尼系数越高，说明注意力越集中（稀疏）
    """
    # 排序
    sorted_weights = np.sort(attention_weights)
    n = len(sorted_weights)
    
    # 计算基尼系数
    cumsum = np.cumsum(sorted_weights)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    return gini

def create_comparison_heatmap(sample_idx, signal, single_attention, signal_len, 
                            confirmed_diseases, probabilities, save_path):
    """
    创建专门用于论文Fig.5对比的Single-Query热力图
    """
    # 🔥 创建与Multi-Query相同格式的对比图
    fig = plt.figure(figsize=(24, 4))  # 与你提供的Multi-Query图相同的宽高比
    
    # 🔥 Single-Query热力图 - 强调稀疏性和奇异性
    ax = fig.add_subplot(1, 1, 1)
    
    # 创建热力图
    attention_2d = single_attention.reshape(1, -1)
    
    # 🔥 使用与Multi-Query不同的颜色方案来突出对比
    im = ax.imshow(attention_2d, cmap='Oranges', aspect='auto', 
                   interpolation='bilinear', extent=[0, signal_len, 0.5, -0.5])
    
    # 🔥 标记稀疏的注意力峰值
    threshold = np.percentile(single_attention, 85)
    peak_indices = np.where(single_attention > threshold)[0]
    
    # 在热力图上突出显示主要的注意力峰值
    if len(peak_indices) > 0:
        # 将连续的峰值区域合并
        peak_regions = []
        current_region = [peak_indices[0]]
        
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] <= 5:  # 如果间隔小于5个时间步，认为是同一区域
                current_region.append(peak_indices[i])
            else:
                peak_regions.append(current_region)
                current_region = [peak_indices[i]]
        peak_regions.append(current_region)
        
        # 在每个峰值区域上方添加标注
        for i, region in enumerate(peak_regions[:3]):  # 最多显示3个主要区域
            center = np.mean(region)
            ax.annotate(f'Peak {i+1}', xy=(center, 0), xytext=(center, 0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=12, fontweight='bold', color='red',
                       ha='center')
    
    # 🔥 计算并显示稀疏性指标
    sparsity_ratio = np.sum(single_attention > threshold) / len(single_attention)
    gini_coeff = calculate_gini_coefficient(single_attention)  # 基尼系数衡量不平等性
    
    # 设置标题 - 突出Single-Query的特征
    active_diseases = [name for _, name in confirmed_diseases]
    title = f'Single-Query Attention Heatmap (Sparse & Singular)\n'
    title += f'Active: {", ".join(active_diseases)} | '
    title += f'Sparsity Ratio: {sparsity_ratio:.3f} | Gini: {gini_coeff:.3f}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Single Query\nAttention', fontsize=12)
    ax.set_yticks([0])
    ax.set_yticklabels(['Shared'])
    ax.set_xlim(0, signal_len)
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.2, shrink=0.8, aspect=50)
    cbar.set_label('Single-Query Attention Weight', fontsize=12)
    
    # 保存对比用的热力图
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
               pad_inches=0.3)
    plt.close()
    
    print(f"✅ 已保存Fig.5对比用Single-Query热力图: {save_path}")

def match_sample_criteria(sample_info, sample_target, classes):
    """检查样本是否符合指定条件"""
    
    # 1. 🔥 按全局索引指定
    if 'global_indices' in sample_target:
        return sample_info['global_idx'] in sample_target['global_indices']
    
    # 2. 🔥 按疾病组合指定
    if 'diseases' in sample_target:
        target_diseases = set(sample_target['diseases'])
        current_diseases = set(sample_info['diseases'])
        
        match_mode = sample_target.get('match_mode', 'exact')  # exact, subset, superset, overlap
        
        if match_mode == 'exact':
            return current_diseases == target_diseases
        elif match_mode == 'subset':
            return target_diseases.issubset(current_diseases)  
        elif match_mode == 'superset':
            return current_diseases.issubset(target_diseases)
        elif match_mode == 'overlap':
            return len(current_diseases & target_diseases) > 0
    
    # 3. 🔥 按疾病数量指定
    if 'num_diseases' in sample_target:
        return sample_info['num_diseases'] == sample_target['num_diseases']
    
    # 4. 🔥 按预测概率范围指定
    if 'prob_range' in sample_target:
        prob_range = sample_target['prob_range']
        max_prob = np.max(sample_info['probabilities'])
        return prob_range[0] <= max_prob <= prob_range[1]
    
    # 5. 🔥 按预测准确性指定  
    if 'prediction_type' in sample_target:
        true_labels = sample_info['true_labels']
        pred_labels = sample_info['pred_labels']
        
        if sample_target['prediction_type'] == 'correct':
            return np.array_equal(true_labels, pred_labels)
        elif sample_target['prediction_type'] == 'incorrect':
            return not np.array_equal(true_labels, pred_labels)
        elif sample_target['prediction_type'] == 'false_positive':
            return np.sum((pred_labels == 1) & (true_labels == 0)) > 0
        elif sample_target['prediction_type'] == 'false_negative':
            return np.sum((pred_labels == 0) & (true_labels == 1)) > 0
    
    return False

def generate_single_query_explanation(model, data_loader, classes, device, output_dir, sample_target=None):
    """
    为Single-Query模型生成可解释性分析，格式与Multi-Query完全一致
    """
    print("🔍 开始生成Single-Query可解释性分析...")
    
    model.eval()
    explanations_dir = output_dir / "explanations"
    explanations_dir.mkdir(exist_ok=True)
    
    # 收集样本进行分析
    collected_samples = []
    sample_count = 0
    
    print(f"🎯 正在搜索目标疾病组合: {sample_target.get('diseases', '未指定') if sample_target else '自动选择'}")
    
    with torch.no_grad():
        for batch_idx, (signals, targets) in enumerate(data_loader):
            signals = signals.to(device, non_blocking=True)
            outputs, attention_weights = model(signals)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            for sub_batch_idx in range(signals.size(0)):
                true_labels = targets[sub_batch_idx].cpu().numpy()
                pred_labels = preds[sub_batch_idx].cpu().numpy()
                prob_values = probs[sub_batch_idx].cpu().numpy()
                
                # 🔥 构建当前样本信息
                current_diseases = [classes[i] for i, label in enumerate(true_labels) if label == 1]
                sample_info = {
                    'batch_idx': batch_idx,
                    'sub_batch_idx': sub_batch_idx,
                    'global_idx': sample_count,  # 全局索引
                    'signal': signals[sub_batch_idx].cpu().numpy(),
                    'attention_weights': attention_weights[sub_batch_idx].cpu().numpy(),
                    'true_labels': true_labels,
                    'pred_labels': pred_labels,
                    'probabilities': prob_values,
                    'diseases': current_diseases,
                    'num_diseases': len(current_diseases)
                }
                
                # 🔥 检查是否匹配指定条件
                if sample_target is not None:
                    if match_sample_criteria(sample_info, sample_target, classes):
                        collected_samples.append(sample_info)
                        print(f"✅ 找到匹配样本 #{len(collected_samples)} (全局索引: {sample_count})")
                        print(f"   - 确诊疾病: {', '.join(current_diseases)}")
                        print(f"   - 预测概率: {[f'{classes[i]}={prob_values[i]:.3f}' for i in range(len(classes))]}")
                        
                        # 🔥 如果找到了完全匹配的样本，优先使用它
                        target_diseases = set(sample_target.get('diseases', []))
                        if set(current_diseases) == target_diseases:
                            print(f"🎯 找到完美匹配的样本！疾病组合: {', '.join(current_diseases)}")
                else:
                    # 🔥 默认收集策略：多标签样本
                    if len(current_diseases) > 1:
                        collected_samples.append(sample_info)
                        print(f"📝 收集多标签样本 #{len(collected_samples)} (全局索引: {sample_count})")
                        print(f"   - 疾病: {', '.join(current_diseases)}")
                
                sample_count += 1
                
                # 🔥 达到目标数量就停止
                target_num = sample_target.get('num_samples', 3) if sample_target else 3
                if len(collected_samples) >= target_num:
                    break
            
            if len(collected_samples) >= target_num:
                break
    
    if not collected_samples:
        print("⚠️ 未找到符合条件的样本进行分析")
        print(f"💡 建议:")
        print(f"   1. 检查目标疾病名称是否正确: {classes.tolist()}")
        print(f"   2. 尝试使用 --match_mode overlap 来放宽匹配条件")
        return
    
    print(f"\n🎯 总共收集到 {len(collected_samples)} 个样本进行可解释性分析")
    
    # 🔥 生成与Multi-Query完全一致格式的可解释性图
    for sample_idx, sample in enumerate(collected_samples[:3]):
        print(f"🎨 正在生成Single-Query样本 {sample_idx + 1} 的可解释性分析...")
        
        signal = sample['signal'][:3]  # 前3个导联
        attention_weights = sample['attention_weights']  # (num_classes, seq_len)
        true_labels = sample['true_labels']
        pred_labels = sample['pred_labels']
        probabilities = sample['probabilities']
        signal_len = signal.shape[1]
        
        # 🔥 Single-Query的关键特征：所有类别共享相同的注意力权重
        single_attention = attention_weights[0]  # (seq_len,)
        
        # 识别确诊的疾病
        confirmed_diseases = [(i, classes[i]) for i, label in enumerate(true_labels) if label == 1]
        if len(confirmed_diseases) == 0:
            top_prob_indices = np.argsort(probabilities)[-2:]
            confirmed_diseases = [(i, classes[i]) for i in top_prob_indices]
        
        # 🔥 创建与Multi-Query完全一致的布局
        num_confirmed = len(confirmed_diseases)
        total_rows = 1 + num_confirmed + 1  # ECG + 确诊疾病热力图 + 组合热力图
        
        # 🔥 使用与Multi-Query完全相同的图形设置
        fig = plt.figure(figsize=(24, total_rows * 3.5))
        gs = fig.add_gridspec(total_rows, 1, 
                             height_ratios=[1.2] + [0.8] * num_confirmed + [1.0],
                             hspace=0.8)
        
        # 1. 原始ECG信号 - 与Multi-Query完全一致
        ax1 = fig.add_subplot(gs[0])
        for i in range(3):
            ax1.plot(signal[i], label=f'Lead {i+1}', alpha=0.8, linewidth=1.5)
        ax1.set_title(f'Sample {sample_idx + 1} - Original ECG Signal (First 3 Leads)', 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Time Steps', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, signal_len)
        
        # 2. 🔥 为每个确诊疾病单独绘制热力图 - 使用Single-Query的注意力
        for heat_idx, (class_idx, class_name) in enumerate(confirmed_diseases):
            ax_heat = fig.add_subplot(gs[1 + heat_idx])
            
            prob = probabilities[class_idx]
            true_label = true_labels[class_idx]
            pred_label = pred_labels[class_idx]
            
            # 🔥 Single-Query关键：所有疾病都使用相同的注意力权重
            attention_upsampled = np.interp(
                np.linspace(0, len(single_attention)-1, signal_len),
                np.arange(len(single_attention)),
                single_attention
            )
            
            # 平滑处理 - 与Multi-Query一致
            attention_smooth = gaussian_filter1d(attention_upsampled, sigma=1.5)
            
            # 🔥 使用与Multi-Query完全相同的viridis颜色方案
            attention_2d = attention_smooth.reshape(1, -1)
            
            im = ax_heat.imshow(attention_2d, cmap='viridis', aspect='auto', 
                               interpolation='bilinear', extent=[0, signal_len, 0.5, -0.5])
            
            # 设置状态指示 - 与Multi-Query完全一致
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
            
            # 🔥 标题格式与Multi-Query完全一致，但标注为Single-Query
            title = f'{class_name} Attention Heatmap | P={prob:.3f} | True={int(true_label)} | Pred={int(pred_label)} | {status}'
            ax_heat.set_title(title, fontsize=14, fontweight='bold', color=title_color, pad=15)
            ax_heat.set_xlabel('Time Steps', fontsize=12)
            ax_heat.set_ylabel('Attention', fontsize=12)
            ax_heat.set_yticks([0])
            ax_heat.set_yticklabels(['Weight'])
            ax_heat.set_xlim(0, signal_len)
            
            # colorbar设置与Multi-Query一致
            cbar = plt.colorbar(im, ax=ax_heat, orientation='horizontal', 
                               pad=0.4, shrink=0.6, aspect=30)
            cbar.set_label('Attention Weight', fontsize=10)
        
        # 3. 🔥 组合热力图 - Single-Query版本
        ax_combined = fig.add_subplot(gs[1 + num_confirmed])
        
        # 🔥 Single-Query的组合注意力：所有疾病共享同一个注意力模式
        combined_attention = np.zeros(signal_len)
        active_diseases_info = []
        
        # 获取确诊疾病的总概率
        total_prob = sum(probabilities[class_idx] for class_idx, _ in confirmed_diseases)
        
        if total_prob > 0:
            # 使用概率加权的单一注意力模式
            attention_upsampled = np.interp(
                np.linspace(0, len(single_attention)-1, signal_len),
                np.arange(len(single_attention)),
                single_attention
            )
            combined_attention = gaussian_filter1d(attention_upsampled, sigma=2.0) * total_prob
        
        for class_idx, class_name in confirmed_diseases:
            prob = probabilities[class_idx]
            active_diseases_info.append(f"{class_name}(P={prob:.2f})")
        
        # 归一化
        if combined_attention.max() > 0:
            combined_attention = combined_attention / combined_attention.max()
        
        # 🔥 绘制组合热力图 - 使用viridis颜色方案与Multi-Query一致
        combined_2d = combined_attention.reshape(1, -1)
        im_combined = ax_combined.imshow(combined_2d, cmap='viridis', aspect='auto', 
                                        interpolation='bilinear', extent=[0, signal_len, 0.5, -0.5])
        
        # 🔥 标题格式与Multi-Query一致，但标注为Single-Query特征
        combined_title = f'Probability-Weighted Combined Attention (Confirmed Diseases)\nActive: {", ".join(active_diseases_info)}'
        ax_combined.set_title(combined_title, fontsize=16, fontweight='bold', pad=20)
        ax_combined.set_xlabel('Time Steps', fontsize=14)
        ax_combined.set_ylabel('Combined\nAttention', fontsize=12)
        ax_combined.set_yticks([0])
        ax_combined.set_yticklabels(['Weights'])
        ax_combined.set_xlim(0, signal_len)
        
        # colorbar设置与Multi-Query一致
        cbar_combined = plt.colorbar(im_combined, ax=ax_combined, orientation='horizontal', 
                                    pad=0.3, shrink=0.6, aspect=30)
        cbar_combined.set_label('Combined Attention Weight', fontsize=12)
        
        # 🔥 设置总标题 - 与Multi-Query格式完全一致
        confirmed_names = [name for _, name in confirmed_diseases]
        correct_predictions = np.sum(true_labels == pred_labels)
        
        plt.suptitle(f'Sample {sample_idx + 1} - Single-Query Attention Analysis\n'
                    f'Confirmed Diseases: {", ".join(confirmed_names)} | '
                    f'Correctly Predicted: {correct_predictions}/{len(classes)}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 保存Single-Query可解释性图像
        save_path = explanations_dir / f"single_query_explanation_sample_{sample_idx + 1}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   pad_inches=0.5)
        plt.close()
        
        print(f"✅ 已保存Single-Query样本 {sample_idx + 1} 的可解释性分析图: {save_path}")

def run_explain_only_single_query(args):
    """仅运行Single-Query可解释性分析"""
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    
    # 检查模型文件是否存在
    model_path = Path('/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/models/saved_models/medea_single_query_experiment/fold_1/best_medea_single_query_model.pth')
    if not model_path.exists():
        print(f"❌ 找不到已训练的Single-Query模型文件: {model_path}")
        print(f"💡 建议:")
        print(f"   1. 先运行训练: python MedeA-singlequery.py --fold {args.fold}")
        print(f"   2. 或检查模型文件路径是否正确")
        return
    
    print(f"🔍 启动Single-Query可解释性分析模式 (Fold {args.fold})...")
    print(f"   - 设备: {device}")
    print(f"   - 模型路径: {model_path}")
    
    # 加载数据
    try:
        _, _, test_loader, classes = create_dataloaders_cv(
            args.data_dir, args.fold, args.batch_size, args.num_workers
        )
    except FileNotFoundError as e:
        print(f"❌ 数据文件加载失败: {e}")
        return
    
    # 🔥 显示可用的疾病类别
    print(f"📋 可用的疾病类别: {classes.tolist()}")
    
    # 创建模型并加载权重
    model = MeDeASingleQuery(
        num_classes=len(classes),
        d_model=args.d_model,
        base_filters=args.base_filters,
        dropout=args.dropout
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Single-Query模型权重加载完成")
    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        return
    
    # 🔥 构建样本筛选条件 - 专门寻找HYP、MI、STTC的组合
    sample_target = None
    
    # 🔥 根据命令行参数构建筛选条件
    if hasattr(args, 'target_diseases') and args.target_diseases:
        sample_target = {
            'diseases': args.target_diseases,
            'match_mode': getattr(args, 'match_mode', 'exact'),
            'num_samples': getattr(args, 'num_samples', 3)
        }
        print(f"🎯 目标疾病: {args.target_diseases}")
        print(f"🎯 匹配模式: {sample_target['match_mode']}")
    else:
        # 🔥 默认设置：寻找HYP、MI、STTC的组合
        sample_target = {
            'diseases': ['HYP', 'MI', 'STTC'],
            'match_mode': 'exact',  # 精确匹配这三种疾病
            'num_samples': 3
        }
        print(f"🎯 默认目标疾病组合: HYP, MI, STTC")
    
    # 运行可解释性分析
    generate_single_query_explanation(
        model=model,
        data_loader=test_loader,
        classes=classes,
        device=device,
        output_dir=output_dir,
        sample_target=sample_target
    )
    
    print(f"🎉 Single-Query可解释性分析完成！")
    print(f"📁 结果保存在: {output_dir}/explanations/")

# ==============================================================================
# 6. 主程序入口 - 添加新的参数
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🚀 MeDeA Single-Query 模型训练脚本')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data', help='包含十折交叉验证数据的目录路径')
    parser.add_argument('--output_dir', type=str, default='./saved_models/medea_single_query_experiment', help='保存模型和结果的目录')
    
    # 交叉验证相关参数
    parser.add_argument('--cross_validation', action='store_true', help='运行十折交叉验证')
    parser.add_argument('--fold', type=int, default=1, help='指定训练单个fold (1-10)')
    parser.add_argument('--verbose', action='store_true', help='显示详细的训练过程')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='注意力头的隐藏维度')
    parser.add_argument('--base_filters', type=int, default=64, help='CNN骨干网络的基础滤波器数量')
    parser.add_argument('--dropout', type=float, default=0.3)

    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    
    # 🔥 可解释性分析的新参数
    parser.add_argument('--explain_only', action='store_true', 
                       help='仅运行Single-Query可解释性分析，不进行训练')
    
    # 🔥 样本选择参数
    parser.add_argument('--target_diseases', nargs='+', type=str, default=None,
                       help='指定目标疾病名称列表，例如: --target_diseases "HYP" "MI" "STTC"')
    
    parser.add_argument('--match_mode', type=str, default='exact',
                       choices=['exact', 'subset', 'superset', 'overlap'],
                       help='疾病匹配模式: exact=完全匹配, subset=目标是子集, superset=目标是超集, overlap=有交集')
    
    parser.add_argument('--target_indices', nargs='+', type=int, default=None,
                       help='指定目标样本的全局索引，例如: --target_indices 0 15 27')
    
    parser.add_argument('--num_diseases', type=int, default=None,
                       help='指定目标样本的疾病数量，例如: --num_diseases 2 (找多标签样本)')
    
    parser.add_argument('--num_samples', type=int, default=3,
                       help='要分析的样本数量 (默认: 3)')
    
    args = parser.parse_args()
    
    if args.explain_only:
        # 🔥 运行Single-Query可解释性分析
        run_explain_only_single_query(args)
    elif args.cross_validation:
        # 运行十折交叉验证
        run_cross_validation(args)
    else:
        # 训练单个fold
        main_training_single_fold(args)