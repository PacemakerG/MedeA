# -*- coding: utf-8 -*-
"""
RNN基线模型 - ECG分类任务

使用LSTM和GRU的循环神经网络架构进行ECG序列分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from tqdm import tqdm
import random

def load_ptbxl_data(data_file_path):
    """加载PTB-XL数据"""
    print(f"正在加载PTB-XL数据: {data_file_path}")
    data = np.load(data_file_path, allow_pickle=True)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    
    print(f"✅ 数据形状:")
    print(f"   - 训练集: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"   - 验证集: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"   - 测试集: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, classes

class RNNBaseline(nn.Module):
    """RNN基线模型 (LSTM + GRU)"""
    
    def __init__(self, input_channels=12, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3, rnn_type='lstm'):
        super(RNNBaseline, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # RNN层
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        batch_size, channels, seq_len = x.size()
        
        # 转置为 (batch_size, sequence_length, channels)
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        # RNN处理
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size * 2)
        
        # 注意力机制
        attention_weights = self.attention(rnn_out)  # (batch_size, seq_len, 1)
        attended_output = torch.sum(rnn_out * attention_weights, dim=1)  # (batch_size, hidden_size * 2)
        
        # 分类
        output = self.classifier(attended_output)
        return output

def find_optimal_thresholds(model, val_loader, device):
    """寻找最优阈值"""
    model.eval()
    val_probs = []
    val_true = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output)
            val_probs.append(probs.cpu().numpy())
            val_true.append(target.cpu().numpy())
    
    val_probs = np.concatenate(val_probs)
    val_true = np.concatenate(val_true)
    
    num_classes = val_true.shape[1]
    optimal_thresholds = np.zeros(num_classes)
    
    print("\n🔥 正在为每个类别寻找最优F1阈值...")
    for i in range(num_classes):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.01):
            preds = (val_probs[:, i] > thresh).astype(int)
            f1 = f1_score(val_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds[i] = best_thresh
    
    print(f"✅ 最优阈值查找完成: {optimal_thresholds}")
    return torch.tensor(optimal_thresholds).to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience, save_path):
    """训练模型"""
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = (torch.sigmoid(output) > 0.5).float()
                val_preds.append(preds.cpu().numpy())
                val_true.append(target.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}, Val F1-macro: {val_f1:.4f}, Train Loss: {train_loss/len(train_loader):.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"🚀 新的最佳模型已保存，F1-macro: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停: {patience} 个epoch无改善")
            break
    
    return best_f1

def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_ptbxl_data(args.data_file)
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
    
    # 创建DataLoader
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNBaseline(
        input_channels=12,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(classes),
        dropout=args.dropout,
        rnn_type=args.rnn_type
    ).to(device)
    
    print(f"✅ {args.rnn_type.upper()}基线模型已初始化，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 训练模型
    print(f"\n🔥 开始训练{args.rnn_type.upper()}基线模型")
    best_f1 = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args.epochs, args.patience, args.save_path
    )
    
    # 测试模型
    print(f"\n🔥 在测试集上评估最佳{args.rnn_type.upper()}模型")
    model.load_state_dict(torch.load(args.save_path, weights_only=True))
    
    # 寻找最优阈值
    optimal_thresholds = find_optimal_thresholds(model, val_loader, device)
    
    # 测试
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="测试中"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = (torch.sigmoid(output) > optimal_thresholds).float()
            test_preds.append(preds.cpu().numpy())
            test_true.append(target.cpu().numpy())
    
    test_preds = np.concatenate(test_preds)
    test_true = np.concatenate(test_true)
    
    print(f"\n--- {args.rnn_type.upper()}基线模型分类报告 (测试集 @ 最优阈值) ---")
    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
    
    # 计算总体指标
    macro_f1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(test_true, test_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    
    print(f"\n📊 {args.rnn_type.upper()}基线模型总体性能指标:")
    print(f"   - Macro F1: {macro_f1:.4f}")
    print(f"   - Micro F1: {micro_f1:.4f}")
    print(f"   - Weighted F1: {weighted_f1:.4f}")
    
    # 保存结果
    results = {
        'model_type': f'{args.rnn_type.upper()}_Baseline',
        'test_predictions': test_preds,
        'test_true': test_true,
        'optimal_thresholds': optimal_thresholds.cpu().numpy(),
        'best_val_f1': best_f1,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1
    }
    np.save(args.results_path, results)
    print(f"✅ 结果已保存到: {args.results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNN基线模型训练')
    parser.add_argument('--data_file', type=str, default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data/ptbxl_processed_100hz.npz', help='数据文件路径')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], help='RNN类型')
    parser.add_argument('--hidden_size', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='RNN层数')
    parser.add_argument('--epochs', type=int, default=30, help='训练周期数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_path', type=str, default='./saved_models/rnn_baseline_model.pth', help='模型保存路径')
    parser.add_argument('--results_path', type=str, default='./results/rnn_baseline_results.npy', help='结果保存路径')
    args = parser.parse_args()
    main(args)
