# -*- coding: utf-8 -*-
"""
Transformer基线模型 - ECG分类任务

使用标准的Transformer架构进行ECG序列分类
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
import math

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

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBaseline(nn.Module):
    """Transformer基线模型"""
    
    def __init__(self, input_channels=12, d_model=128, nhead=8, num_layers=6, num_classes=5, dropout=0.1, max_seq_len=1000):
        super(TransformerBaseline, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        batch_size, channels, seq_len = x.size()
        
        # 转置为 (batch_size, sequence_length, channels)
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 全局池化
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(x)
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
    model = TransformerBaseline(
        input_channels=12,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_classes=len(classes),
        dropout=args.dropout,
        max_seq_len=1000
    ).to(device)
    
    print(f"✅ Transformer基线模型已初始化，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    
    # 训练模型
    print(f"\n🔥 开始训练Transformer基线模型")
    best_f1 = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args.epochs, args.patience, args.save_path
    )
    
    # 检查模型文件是否存在，如果不存在则保存当前模型
    if not os.path.exists(args.save_path):
        print(f"⚠️ 未找到最佳模型文件，保存当前模型状态")
        torch.save(model.state_dict(), args.save_path)
        best_f1 = 0.0  # 如果没有保存过最佳模型，说明验证性能很差
    
    # 测试模型
    print(f"\n🔥 在测试集上评估最佳Transformer模型")
    
    # 安全地加载模型
    try:
        print(f"正在加载模型: {args.save_path}")
        model.load_state_dict(torch.load(args.save_path, weights_only=True))
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print(f"使用当前训练的模型进行测试")
    
    # 寻找最优阈值
    print(f"\n🔍 寻找最优阈值...")
    optimal_thresholds = find_optimal_thresholds(model, val_loader, device)
    
    # 测试
    print(f"\n🧪 在测试集上进行最终评估...")
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
    
    print(f"\n--- Transformer基线模型分类报告 (测试集 @ 最优阈值) ---")
    print(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
    
    # 计算总体指标
    macro_f1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(test_true, test_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    
    print(f"\n📊 Transformer基线模型总体性能指标:")
    print(f"   - Macro F1: {macro_f1:.4f}")
    print(f"   - Micro F1: {micro_f1:.4f}")
    print(f"   - Weighted F1: {weighted_f1:.4f}")
    print(f"   - Best Validation F1: {best_f1:.4f}")
    
    # 保存结果
    results = {
        'model_type': 'Transformer_Baseline',
        'test_predictions': test_preds,
        'test_true': test_true,
        'optimal_thresholds': optimal_thresholds.cpu().numpy(),
        'best_val_f1': best_f1,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'model_params': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr
        }
    }
    
    try:
        np.save(args.results_path, results)
        print(f"✅ 结果已保存到: {args.results_path}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
    
    print(f"\n✅ Transformer基线模型训练完成!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer基线模型训练')
    parser.add_argument('--data_file', type=str, default='/home/elonge/WorkSpace/ECG_Project/processed_data/ptbxl_processed_100hz.npz', help='数据文件路径')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--epochs', type=int, default=30, help='训练周期数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_path', type=str, default='./saved_models/transformer_baseline_model.pth', help='模型保存路径')
    parser.add_argument('--results_path', type=str, default='./results/transformer_baseline_results.npy', help='结果保存路径')
    args = parser.parse_args()
    main(args)
