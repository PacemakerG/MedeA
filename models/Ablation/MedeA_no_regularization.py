# -*- coding: utf-8 -*-
"""
MeDeA 消融实验 - 无正则化版本 (w/o Regularization)
移除dropout、batch normalization等正则化技术，测试正则化对性能的影响
对应表格: (G) Backbone + Multi-Query + w/o Regularization
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

class CNNFeatureExtractorNoReg(nn.Module):
    """🔥 无正则化的CNN特征提取器 - 移除所有正则化技术"""
    def __init__(self, input_channels=12, base_filters=64):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, base_filters, kernel_size=7, padding=3),
                # 🔥 移除 BatchNorm1d
                nn.ReLU(),
                nn.MaxPool1d(2),
                # 🔥 移除 Dropout1d
            ),
            nn.Sequential(
                nn.Conv1d(base_filters, base_filters*2, kernel_size=5, padding=2),
                # 🔥 移除 BatchNorm1d
                nn.ReLU(),
                nn.MaxPool1d(2),
                # 🔥 移除 Dropout1d
            ),
            nn.Sequential(
                nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
                # 🔥 移除 BatchNorm1d
                nn.ReLU(),
                nn.MaxPool1d(2),
                # 🔥 移除 Dropout1d
            ),
            nn.Sequential(
                nn.Conv1d(base_filters*4, base_filters*8, kernel_size=3, padding=1),
                # 🔥 移除 BatchNorm1d
                nn.ReLU(),
                nn.MaxPool1d(2),
                # 🔥 移除 Dropout1d
            )
        ])
        
        self.output_dim = base_filters * 8
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x.transpose(1, 2)  # (B, T, C)

class AttentionHeadNoReg(nn.Module):
    """🔥 无正则化的注意力头 - 移除dropout"""
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.query = nn.Linear(d_model, d_k, bias=False)
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_k, bias=False)
        # 🔥 移除 dropout
        
    def forward(self, q_input, kv_input, mask=None):
        q = self.query(q_input)
        k = self.key(kv_input)
        v = self.value(kv_input)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        # 🔥 不应用 dropout
        
        attended_values = torch.matmul(attention_weights, v)
        return attended_values, attention_weights

class MultiHeadQueryAttentionNoReg(nn.Module):
    """🔥 无正则化的多疾病查询注意力模块"""
    def __init__(self, d_model, num_heads, num_diseases):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_diseases = num_diseases
        self.d_k = d_model // num_heads
        
        # 多疾病查询向量
        self.disease_queries = nn.Parameter(torch.randn(num_diseases, d_model))
        
        # 多头注意力（无正则化）
        self.attention_heads = nn.ModuleList([
            AttentionHeadNoReg(d_model, self.d_k) for _ in range(num_heads)
        ])
        
        self.output_projection = nn.Linear(d_model, d_model)
        # 🔥 移除 dropout
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 扩展疾病查询到batch维度
        disease_queries = self.disease_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        all_attended_values = []
        all_attention_weights = []
        
        for head in self.attention_heads:
            attended_values, attention_weights = head(disease_queries, x)
            all_attended_values.append(attended_values)
            all_attention_weights.append(attention_weights)
        
        # 拼接多头输出
        multi_head_output = torch.cat(all_attended_values, dim=-1)
        output = self.output_projection(multi_head_output)
        # 🔥 不应用 dropout
        
        # 平均多头注意力权重
        avg_attention = torch.stack(all_attention_weights, dim=0).mean(dim=0)
        
        return output, avg_attention

# 🔥 消融实验模型：无正则化版本
class MeDeANoRegularization(nn.Module):
    """消融实验：移除所有正则化技术的模型"""
    def __init__(self, num_classes=5, d_model=256, num_heads=8, base_filters=64):
        super().__init__()
        
        # 🔥 无正则化的CNN特征提取器
        self.feature_extractor = CNNFeatureExtractorNoReg(
            input_channels=12, 
            base_filters=base_filters
        )
        
        # 特征维度映射（无正则化）
        self.feature_proj = nn.Linear(self.feature_extractor.output_dim, d_model)
        
        # 🔥 无正则化的多疾病注意力模块
        self.multi_query_attention = MultiHeadQueryAttentionNoReg(
            d_model=d_model,
            num_heads=num_heads,
            num_diseases=num_classes
        )
        
        # 分类层
        self.classifier = nn.Linear(d_model, 1)
        
        # 🔥 移除 Layer Normalization 和 Dropout
        
    def forward(self, x):
        # CNN特征提取（无正则化）
        features = self.feature_extractor(x)
        
        # 特征投影（无正则化）
        features = self.feature_proj(features)
        # 🔥 不应用 layer norm 和 dropout
        
        # 无正则化的多疾病查询注意力
        disease_representations, attention_weights = self.multi_query_attention(features)
        
        # 🔥 不应用 layer norm 和 dropout
        
        # 分类预测
        logits = self.classifier(disease_representations).squeeze(-1)
        
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
    model = MeDeANoRegularization(
        num_classes=len(classes),
        d_model=args.d_model,
        num_heads=args.num_heads,
        base_filters=args.base_filters
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    # 🔥 不使用weight_decay（权重衰减也是正则化技术）
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    best_f1, patience_counter = 0.0, 0
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for signals, targets in train_loader:
            signals, targets = signals.to(device), targets.to(device)
            
            outputs, attention_weights = model(signals)
            
            # 🔥 仅使用分类损失，不使用注意力正则化
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        val_f1, val_auc = run_evaluation(model, val_loader, device)
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_no_reg_fold_{fold}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            break
    
    # 测试评估
    model.load_state_dict(torch.load(f'best_no_reg_fold_{fold}.pth'))
    test_f1, test_auc = run_evaluation(model, test_loader, device)
    
    # 清理临时文件
    os.remove(f'best_no_reg_fold_{fold}.pth')
    
    return test_f1, test_auc

def main():
    parser = argparse.ArgumentParser(description='MeDeA Ablation: No Regularization')
    parser.add_argument('--data_dir', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_folds', type=int, default=10)
    
    args = parser.parse_args()
    
    f1_scores = []
    auc_scores = []
    
    print("🚀 开始消融实验: 无正则化版本")
    
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
    
    print(f"\n📊 消融实验结果 (w/o Regularization):")
    print(f"Macro F1:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Macro AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # 保存结果
    results = {
        'experiment': 'w/o Regularization',
        'description': 'Backbone + Multi-Query + w/o Regularization',
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'individual_folds': {
            'f1_scores': f1_scores,
            'auc_scores': auc_scores
        }
    }
    
    with open('ablation_no_regularization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return mean_f1, mean_auc

if __name__ == "__main__":
    main()