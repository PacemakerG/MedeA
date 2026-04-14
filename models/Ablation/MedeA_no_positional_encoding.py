# -*- coding: utf-8 -*-
"""
MeDeA 消融实验 - 无学习查询版本 (w/o Learned Query)
使用固定查询向量而非学习的查询向量
对应表格: (B) Backbone + Fixed Single-Query
"""
import os
import argparse
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import math

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PTBXLDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def create_dataloaders_cv(data_dir, fold_num, batch_size, num_workers):
    data_file_path = Path(data_dir) / f"ptbxl_processed_100hz_fold{fold_num}.npz"
    if not data_file_path.exists():
        raise FileNotFoundError(f"❌ 数据文件未找到: {data_file_path}")
    
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
    
    return train_loader, val_loader, test_loader, classes

class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器"""
    def __init__(self, input_channels=12, base_filters=64, dropout=0.3):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, base_filters, kernel_size=7, padding=3),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(base_filters, base_filters*2, kernel_size=5, padding=2),
                nn.BatchNorm1d(base_filters*2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_filters*4),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(base_filters*4, base_filters*8, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_filters*8),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout)
            )
        ])
        
        self.output_dim = base_filters * 8
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x.transpose(1, 2)  # (B, T, C)

class FixedSingleQueryAttention(nn.Module):
    """🔥 Fixed Single-Query 注意力模块 - 使用固定的查询向量"""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # 🔥 Fixed Query: 固定的查询向量，不参与学习
        fixed_query = torch.randn(1, 1, d_model)
        self.register_buffer('fixed_query', fixed_query)  # 不参与梯度更新
        
        # Key和Value的投影层
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 分类器
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = d_model ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 🔥 使用固定查询（不学习）
        query = self.fixed_query.expand(batch_size, -1, -1)
        
        # 计算keys和values
        keys = self.key_proj(x)
        values = self.value_proj(x)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, values)
        
        # 输出投影
        output = self.output_proj(attended_values)
        output = output.squeeze(1)
        
        # 分类
        logits = self.classifier(output)
        
        # 扩展注意力权重以保持接口一致性
        expanded_attention = attention_weights.squeeze(1).unsqueeze(1).expand(-1, self.num_classes, -1)
        
        return logits, expanded_attention

# 🔥 消融实验模型：固定单查询版本
class MeDeAFixedSingleQuery(nn.Module):
    """消融实验：使用固定查询向量的单查询模型"""
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
        
        # 🔥 Fixed Single-Query 注意力模块
        self.attention = FixedSingleQueryAttention(d_model, num_classes)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # CNN特征提取
        features = self.feature_extractor(x)
        
        # 特征投影
        features = self.feature_proj(features)
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        # Fixed Single-Query 注意力
        logits, attention_weights = self.attention(features)
        
        return logits, attention_weights

def run_evaluation(model, data_loader, device):
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
        
    return f1_macro, auc_macro

def run_single_fold_ablation(args, fold):
    """运行单个fold的消融实验"""
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_loader, val_loader, test_loader, classes = create_dataloaders_cv(
        args.data_dir, fold, args.batch_size, args.num_workers
    )
    
    # 创建模型
    model = MeDeAFixedSingleQuery(
        num_classes=len(classes),
        d_model=args.d_model,
        base_filters=args.base_filters,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    best_f1, patience_counter = 0.0, 0
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for signals, targets in train_loader:
            signals, targets = signals.to(device), targets.to(device)
            
            outputs, attention_weights = model(signals)
            
            # 主要分类损失
            classification_loss = criterion(outputs, targets)
            
            # 注意力正则化（可选）
            attention_entropy = -torch.sum(
                attention_weights.mean(dim=1) * torch.log(attention_weights.mean(dim=1) + 1e-8), 
                dim=-1
            ).mean()
            
            loss = classification_loss - 0.01 * attention_entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        val_f1, val_auc = run_evaluation(model, val_loader, device)
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_fixed_query_fold_{fold}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            break
    
    # 测试评估
    model.load_state_dict(torch.load(f'best_fixed_query_fold_{fold}.pth'))
    test_f1, test_auc = run_evaluation(model, test_loader, device)
    
    # 清理临时文件
    os.remove(f'best_fixed_query_fold_{fold}.pth')
    
    return test_f1, test_auc

def main():
    parser = argparse.ArgumentParser(description='MeDeA Ablation: Fixed Single Query')
    parser.add_argument('--data_dir', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_folds', type=int, default=10)
    
    args = parser.parse_args()
    
    f1_scores = []
    auc_scores = []
    
    print("🚀 开始消融实验: 固定单查询版本")
    
    for fold in range(1, args.num_folds + 1):
        print(f"正在运行 Fold {fold}/{args.num_folds}")
        try:
            f1, auc = run_single_fold_ablation(args, fold)
            f1_scores.append(f1)
            auc_scores.append(auc)
            print(f"Fold {fold} - F1: {f1:.4f}, AUC: {auc:.4f}")
        except Exception as e:
            print(f"Fold {fold} 失败: {e}")
            continue
    
    # 计算平均结果
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    print(f"\n📊 消融实验结果 (w/o Learned Query):")
    print(f"Macro F1:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Macro AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # 保存结果
    results = {
        'experiment': 'w/o Learned Query',
        'description': 'Backbone + Fixed Single-Query',
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'individual_folds': {
            'f1_scores': f1_scores,
            'auc_scores': auc_scores
        }
    }
    
    with open('ablation_fixed_query_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return mean_f1, mean_auc

if __name__ == "__main__":
    main()