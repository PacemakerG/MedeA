# -*- coding: utf-8 -*-
"""
MeDeA 消融实验 - 无查询模块版本 (w/o Query Module)
移除查询模块，使用CNN特征提取 + 全局平均池化 + 全连接分类
对应表格: (A) Backbone + Global Average Pooling
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

# 复用数据加载和种子设置函数
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

# 🔥 消融实验模型：无查询模块版本
class MeDeANoQueryModule(nn.Module):
    """消融实验：移除查询模块，仅使用CNN + 全局平均池化"""
    def __init__(self, num_classes=5, base_filters=64, dropout=0.3):
        super().__init__()
        
        # CNN 骨干网络 (与原始模型相同)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(12, base_filters, kernel_size=7, padding=3),
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
        
        # 🔥 全局平均池化替代查询模块
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(base_filters * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # CNN特征提取
        for block in self.conv_blocks:
            x = block(x)
        
        # 🔥 全局平均池化
        x = self.global_avg_pool(x)  # (B, C, 1)
        x = x.squeeze(-1)  # (B, C)
        
        # 分类
        logits = self.classifier(x)
        
        # 返回虚拟注意力权重以保持接口一致性
        batch_size = x.size(0)
        dummy_attention = torch.ones(batch_size, 5, 32).to(x.device) / 32  # 均匀分布
        
        return logits, dummy_attention

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
    model = MeDeANoQueryModule(
        num_classes=len(classes),
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
            
            outputs, _ = model(signals)
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
            torch.save(model.state_dict(), f'best_no_query_fold_{fold}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            break
    
    # 测试评估
    model.load_state_dict(torch.load(f'best_no_query_fold_{fold}.pth'))
    test_f1, test_auc = run_evaluation(model, test_loader, device)
    
    # 清理临时文件
    os.remove(f'best_no_query_fold_{fold}.pth')
    
    return test_f1, test_auc

def main():
    parser = argparse.ArgumentParser(description='MeDeA Ablation: No Attention')
    parser.add_argument('--data_dir', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_folds', type=int, default=10)
    
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        print("请确保数据已正确预处理并放置在指定目录下")
        return 0.0, 0.0
    
    # 检查是否有任何fold的数据文件
    data_files_found = False
    for fold in range(1, args.num_folds + 1):
        data_file = Path(args.data_dir) / f"ptbxl_processed_100hz_fold{fold}.npz"
        if data_file.exists():
            data_files_found = True
            break
    
    if not data_files_found:
        print(f"❌ 未找到任何fold的数据文件 (格式: ptbxl_processed_100hz_fold{{X}}.npz)")
        print(f"请检查数据目录: {args.data_dir}")
        return 0.0, 0.0
    
    f1_scores = []
    auc_scores = []
    
    print("🚀 开始消融实验: 无注意力机制版本")
    print(f"数据目录: {args.data_dir}")
    print(f"计划运行 {args.num_folds} 个fold")
    
    for fold in range(1, args.num_folds + 1):
        print(f"\n正在运行 Fold {fold}/{args.num_folds}")
        try:
            f1, auc = run_single_fold_ablation(args, fold)
            f1_scores.append(f1)
            auc_scores.append(auc)
            print(f"Fold {fold} 完成 - F1: {f1:.4f}, AUC: {auc:.4f}")
        except FileNotFoundError as e:
            print(f"⚠️ Fold {fold} 数据文件不存在: {e}")
            continue
        except Exception as e:
            print(f"❌ Fold {fold} 失败: {e}")
            continue
    
    # 检查是否有足够的结果
    if len(f1_scores) == 0:
        print("❌ 没有成功完成任何fold的实验")
        print("请检查数据文件是否存在和格式是否正确")
        # 保存空结果
        results = {
            'experiment': 'w/o Attention',
            'description': 'Backbone Only (无注意力机制)',
            'mean_f1': 0.0,
            'std_f1': 0.0,
            'mean_auc': 0.0,
            'std_auc': 0.0,
            'individual_folds': {
                'f1_scores': [],
                'auc_scores': []
            },
            'error': 'No successful folds completed'
        }
        
        with open('ablation_no_attention_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return 0.0, 0.0
    
    # 计算平均结果
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    print(f"\n📊 消融实验结果 (w/o Attention):")
    print(f"成功完成的fold数: {len(f1_scores)}/{args.num_folds}")
    print(f"Macro F1:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Macro AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # 保存结果
    results = {
        'experiment': 'w/o Attention',
        'description': 'Backbone Only (无注意力机制)',
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'individual_folds': {
            'f1_scores': f1_scores,
            'auc_scores': auc_scores
        },
        'completed_folds': len(f1_scores),
        'total_folds': args.num_folds
    }
    
    with open('ablation_no_attention_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ 结果已保存到 ablation_no_attention_results.json")
    return mean_f1, mean_auc

if __name__ == "__main__":
    main()