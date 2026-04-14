# -*- coding: utf-8 -*-
"""
使用 MeDeA (Multi-disease Decompositional Attention) 模型训练 PTB-XL 数据集 (23分类版本)

基于原始Multi-Query脚本改造，专门针对PTB-XL官方23个子类别进行优化
"""
import os
import argparse
import random
import time
import copy
import json
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, 
    precision_score, recall_score, roc_curve  # 🔥 添加 roc_curve 导入
)
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
# 添加数据增强
class ECGAugmentation:
    """ECG信号数据增强"""
    def __init__(self, noise_std=0.01, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def __call__(self, x):
        if random.random() > 0.5:  # 50%概率应用增强
            # 添加高斯噪声
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
            # 幅度缩放
            scale = random.uniform(*self.scale_range)
            x = x * scale
            
            # 时间平移
            shift = random.randint(-10, 10)
            if shift != 0:
                x = torch.roll(x, shift, dims=-1)
        
        return x

class PTBXLDataset(Dataset):
    """为PTB-XL ECG数据创建PyTorch Dataset。"""
    def __init__(self, X, y, augmentation=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augmentation:
            x = self.augmentation(x)
        return x, y

def create_dataloaders_cv_23classes(data_dir, fold_num, batch_size, num_workers):
    """从十折交叉验证数据文件夹加载指定fold的23分类数据并创建DataLoaders。"""
    # 🔥 修改1: 更新数据文件路径以支持23分类
    data_file_path = Path(data_dir) / f"ptbxl_subclass_processed_100hz_fold{fold_num}.npz"
    
    if not data_file_path.exists():
        raise FileNotFoundError(f"❌ 数据文件未找到: {data_file_path}")
    
    print(f"⌛️ 正在从 {data_file_path} 加载第 {fold_num} 折数据 (23分类)...")
    data = np.load(data_file_path, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    classes = data['classes']
    
    train_ds = PTBXLDataset(X_train, y_train, augmentation=ECGAugmentation())
    val_ds = PTBXLDataset(X_val, y_val)
    test_ds = PTBXLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"✅ 第 {fold_num} 折数据加载完成。类别数量: {len(classes)}")
    print(f"   - 类别: {classes.tolist()}")
    print(f"   - 训练集: {len(train_ds)} 样本")
    print(f"   - 验证集: {len(val_ds)} 样本") 
    print(f"   - 测试集: {len(test_ds)} 样本")
    
    return train_loader, val_loader, test_loader, classes

# ==============================================================================
# 3. MeDeA 23分类模型定义 (Enhanced for 23 classes)
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

class EnhancedSharedBackbone(nn.Module):
    """🔥 修改2: 增强的共享骨干网络 - 针对23分类优化"""
    def __init__(self, input_channels=12, base_filters=64, dropout=0.1):
        super(EnhancedSharedBackbone, self).__init__()
        
        # 🔥 增加网络深度以处理更复杂的23分类任务
        # 初始卷积层
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层 - 增加更多层以提取更丰富的特征
        self.layer1 = self._make_layer(base_filters, base_filters, 3, stride=1, dropout=dropout)  # 增加blocks
        self.layer2 = self._make_layer(base_filters, base_filters*2, 3, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, 4, stride=2, dropout=dropout)  # 增加blocks
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, 3, stride=2, dropout=dropout)
        
        # 🔥 新增第5层以增强特征表示能力
        self.layer5 = self._make_layer(base_filters*8, base_filters*16, 2, stride=2, dropout=dropout)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = base_filters * 16  # 🔥 更新特征维度
        
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
        x = self.layer4(x)
        feature_maps = self.layer5(x)  # 🔥 使用第5层的特征图用于注意力
        
        # 全局平均池化
        pooled_features = self.avgpool(feature_maps)
        pooled_features = pooled_features.squeeze(-1)  # (batch_size, feature_dim)
        
        return pooled_features, feature_maps

class MultiScaleAttentionHead23(nn.Module):
    """🔥 修改3: 针对23分类优化的多尺度注意力头"""
    def __init__(self, feature_dim, d_model, dropout=0.1):
        super(MultiScaleAttentionHead23, self).__init__()
        self.d_model = d_model
        
        # 🔥 多尺度注意力机制 - 针对ECG的不同频率特征
        # 局部注意力 - 关注短期ECG模式
        self.local_attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
        
        # 全局注意力 - 关注长期ECG模式  
        self.global_attention = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        
        # 特征投影层
        self.feature_projection = nn.Linear(feature_dim, d_model)
        
        # 🔥 增强的查询生成 - 为23个类别生成更具表征性的查询
        self.query_generator = nn.Sequential(
            nn.Linear(feature_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 🔥 增强的分类器 - 针对23分类
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),  # 增加dropout以防止过拟合
            nn.Linear(d_model // 2, d_model // 4),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, pooled_features, feature_maps):
        # pooled_features: (batch_size, feature_dim)
        # feature_maps: (batch_size, feature_dim, seq_len)
        
        batch_size, feature_dim, seq_len = feature_maps.shape
        
        # 将特征图重塑为序列格式
        feature_seq = feature_maps.transpose(1, 2)  # (batch_size, seq_len, feature_dim)
        
        # 投影到注意力空间
        projected_features = self.feature_projection(feature_seq)  # (batch_size, seq_len, d_model)
        
        # 生成查询向量
        query = self.query_generator(pooled_features).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # 局部注意力 - 关注短期模式
        local_attended, local_weights = self.local_attention(
            query, projected_features, projected_features
        )  # (batch_size, 1, d_model)
        
        # 全局注意力 - 关注长期模式
        global_attended, global_weights = self.global_attention(
            query, projected_features, projected_features  
        )  # (batch_size, 1, d_model)
        
        # 特征融合
        combined_features = torch.cat([
            local_attended.squeeze(1), 
            global_attended.squeeze(1)
        ], dim=-1)  # (batch_size, d_model * 2)
        
        fused_features = self.fusion(combined_features)  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(fused_features)  # (batch_size, 1)
        
        # 合并注意力权重
        combined_weights = (local_weights + global_weights) / 2  # (batch_size, 1, seq_len)
        
        return output, combined_weights.squeeze(1)  # (batch_size, 1), (batch_size, seq_len)

class MeDeA23Classes(nn.Module):
    """🔥 修改4: MeDeA模型 - 23分类版本"""
    def __init__(self, num_classes=23, d_model=512, base_filters=64, dropout=0.2):  # 🔥 调整默认参数
        super(MeDeA23Classes, self).__init__()      
        self.num_classes = num_classes
        
        # 🔥 使用增强的共享骨干网络
        self.backbone = EnhancedSharedBackbone(
            input_channels=12, 
            base_filters=base_filters, 
            dropout=dropout
        )
        
        # 🔥 为每个疾病创建专门的多尺度注意力头
        self.attention_heads = nn.ModuleList([
            MultiScaleAttentionHead23(self.backbone.feature_dim, d_model, dropout)
            for _ in range(num_classes)
        ])
        
        # 🔥 添加全局特征正则化
        self.feature_regularizer = nn.BatchNorm1d(self.backbone.feature_dim)
        
        # 🔥 类别间交互建模（可选）
        self.inter_class_interaction = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        pooled_features, feature_maps = self.backbone(x)
        
        # 🔥 特征正则化
        pooled_features = self.feature_regularizer(pooled_features)
        
        outputs = []
        attention_weights = []
        
        # 为每个疾病类别计算输出
        for i, attention_head in enumerate(self.attention_heads):
            output, weights = attention_head(pooled_features, feature_maps)
            outputs.append(output)
            attention_weights.append(weights)
        
        # 合并输出
        final_output = torch.cat(outputs, dim=1)  # (batch_size, num_classes)
        attention_weights = torch.stack(attention_weights, dim=1)  # (batch_size, num_classes, seq_len)
        
        return final_output, attention_weights

# ==============================================================================
# 4. 训练与评估逻辑 (Training & Evaluation)
# ==============================================================================
def run_evaluation_detailed_23(model, data_loader, device, classes):
    """🔥 修改5: 针对23分类的详细评估函数"""
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
    
    # 计算宏平均
    macro_f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
    try:
        macro_auc = roc_auc_score(all_targets, all_probs, average='macro')
    except ValueError:
        macro_auc = 0.0
        
    return macro_f1, macro_auc, all_targets, preds, per_class_metrics

def optimize_thresholds(y_true, y_probs):
    """为每个类别优化分类阈值"""
    optimal_thresholds = []
    
    for i in range(y_true.shape[1]):
        y_true_class = y_true[:, i]
        y_prob_class = y_probs[:, i]
        
        # 使用Youden's J statistic寻找最优阈值
        fpr, tpr, thresholds = roc_curve(y_true_class, y_prob_class)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        optimal_thresholds.append(optimal_threshold)
    
    return np.array(optimal_thresholds)

def run_evaluation_with_threshold_optimization(model, data_loader, device, classes):
    """使用阈值优化的评估函数"""
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
    
    # 🔥 优化阈值
    optimal_thresholds = optimize_thresholds(all_targets, all_probs)
    
    # 使用优化的阈值进行预测
    preds = (all_probs > optimal_thresholds[np.newaxis, :]).astype(int)
    
    # 计算性能指标
    macro_f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
    
    return macro_f1, optimal_thresholds, all_targets, preds

# ==============================================================================
# 5. 23分类交叉验证训练函数
# ==============================================================================
def run_cross_validation_23classes(args):
    """🔥 修改6: 针对23分类的十折交叉验证函数"""
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动 MeDeA 23分类模型十折交叉验证...")
    print(f"   - 设备: {device}")
    print(f"   - 结果将保存至: {output_dir}")
    print(f"   - 从第 {args.start_fold} 折开始")  # 🔥 新增
    
    # 存储所有fold的结果
    all_fold_results = {
        'val_f1': [],
        'val_auc': [],
        'test_f1': [],
        'test_auc': [],
        'fold_times': [],
        'per_class_metrics': []
    }
    
    # 🔥 新增：尝试加载已有的结果
    existing_results_path = output_dir / 'cross_validation_summary_23classes.json'
    if existing_results_path.exists() and args.start_fold > 1:
        print(f"📂 发现现有结果文件，正在加载前 {args.start_fold - 1} 折的结果...")
        try:
            with open(existing_results_path, 'r') as f:
                existing_data = json.load(f)
            
            existing_results = existing_data.get('cross_validation_results', {})
            
            # 加载已完成fold的结果
            completed_folds = len(existing_results.get('test_f1', []))
            if completed_folds >= args.start_fold - 1:
                all_fold_results['val_f1'] = existing_results['val_f1'][:args.start_fold - 1]
                all_fold_results['val_auc'] = existing_results['val_auc'][:args.start_fold - 1]
                all_fold_results['test_f1'] = existing_results['test_f1'][:args.start_fold - 1]
                all_fold_results['test_auc'] = existing_results['test_auc'][:args.start_fold - 1]
                all_fold_results['fold_times'] = existing_results['fold_times'][:args.start_fold - 1]
                all_fold_results['per_class_metrics'] = existing_results['per_class_metrics'][:args.start_fold - 1]
                
                print(f"✅ 已加载前 {len(all_fold_results['test_f1'])} 折的结果")
                for i, (f1, auc, t) in enumerate(zip(
                    all_fold_results['test_f1'], 
                    all_fold_results['test_auc'], 
                    all_fold_results['fold_times']
                ), 1):
                    print(f"   - Fold {i}: F1={f1:.4f}, AUC={auc:.4f}, Time={t/60:.1f}min")
        except Exception as e:
            print(f"⚠️ 加载现有结果时出错: {e}")
            print("   继续进行全新的交叉验证...")
    
    classes = None
    
    # 🔥 修改：从指定fold开始运行
    for fold in range(args.start_fold, 11):  # 从start_fold到fold10
        print(f"\n{'='*60}")
        print(f"🔄 开始第 {fold}/10 折交叉验证 (23分类)")
        print(f"{'='*60}")
        
        fold_start_time = time.time()
        
        # 为当前fold创建专用输出目录
        fold_output_dir = output_dir / f"fold_{fold}"
        fold_output_dir.mkdir(exist_ok=True)
        
        # 🔥 检查该fold是否已完成
        fold_model_path = fold_output_dir / 'best_medea_23classes_model.pth'
        fold_report_path = fold_output_dir / 'classification_report_23classes.txt'
        
        if fold_model_path.exists() and fold_report_path.exists():
            print(f"✅ Fold {fold} 已完成，跳过训练...")
            # 可以选择加载已有结果或重新训练
            continue
        
        # 加载当前fold的23分类数据
        try:
            train_loader, val_loader, test_loader, fold_classes = create_dataloaders_cv_23classes(
                args.data_dir, fold, args.batch_size, args.num_workers
            )
            if classes is None:
                classes = fold_classes
                print(f"📊 确认类别数量: {len(classes)} (目标: 23)")
        except FileNotFoundError as e:
            print(f"⚠️ 跳过第 {fold} 折: {e}")
            continue
        
        # 🔥 创建23分类模型
        model = MeDeA23Classes(
            num_classes=len(classes),
            d_model=args.d_model,
            base_filters=args.base_filters,
            dropout=args.dropout
        ).to(device)
        
        if fold == args.start_fold:  # 🔥 修改：在开始fold显示模型信息
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"🏥 MeDeA 23分类模型配置:")
            print(f"   - 类别数量: {len(classes)}")
            print(f"   - d_model: {args.d_model}")
            print(f"   - base_filters: {args.base_filters}")
            print(f"   - 可训练参数: {param_count:,}")

        # 🔥 针对23分类调整损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        
        # 🔥 使用较小的学习率和权重衰减以适应更复杂的模型
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr * 0.8,  # 降低学习率
            weight_decay=args.weight_decay * 1.2  # 增加权重衰减
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', patience=8, factor=0.6, min_lr=1e-6
        )
        
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
                
                # 🔥 针对23分类的增强正则化
                # 注意力分布正则化
                attention_entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-8), 
                    dim=-1
                ).mean()
                
                # 🔥 类别平衡损失 - 处理23分类中的类别不平衡
                pos_weight = targets.sum(dim=0) + 1  # 避免除零
                neg_weight = (targets.shape[0] - targets.sum(dim=0)) + 1
                class_balance_weight = neg_weight / pos_weight
                
                # 重加权的分类损失
                weighted_classification_loss = F.binary_cross_entropy_with_logits(
                    outputs, targets, 
                    pos_weight=class_balance_weight.to(device)
                )
                
                # 组合损失
                total_loss_batch = (
                    0.8 * weighted_classification_loss +  # 主损失
                    0.1 * classification_loss +  # 标准损失
                    0.1 * (-0.01 * attention_entropy)  # 注意力正则化
                )
                
                optimizer.zero_grad()
                total_loss_batch.backward()
                
                # 🔥 梯度裁剪以提高训练稳定性
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                progress_bar.set_postfix({
                    'cls_loss': f'{classification_loss.item():.4f}',
                    'w_cls_loss': f'{weighted_classification_loss.item():.4f}',
                    'att_ent': f'{attention_entropy.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

            val_f1, val_auc, _, _, _ = run_evaluation_detailed_23(model, val_loader, device, classes)
            scheduler.step(val_f1)
            
            if args.verbose or epoch % 10 == 0:
                print(f"Fold {fold} Epoch {epoch+1}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Loss: {total_loss / len(train_loader):.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), fold_output_dir / 'best_medea_23classes_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"⌛️ Fold {fold} 早停: 在第 {epoch+1} 个epoch触发")
                break
        
        # 在测试集上评估最终模型
        model.load_state_dict(torch.load(fold_output_dir / 'best_medea_23classes_model.pth', map_location=device))
        test_f1, test_auc, test_true, test_preds, per_class_metrics = run_evaluation_detailed_23(model, test_loader, device, classes)
        
        fold_time = time.time() - fold_start_time
        
        # 保存当前fold的结果
        all_fold_results['val_f1'].append(best_f1)
        all_fold_results['val_auc'].append(val_auc)
        all_fold_results['test_f1'].append(test_f1)
        all_fold_results['test_auc'].append(test_auc)
        all_fold_results['fold_times'].append(fold_time)
        all_fold_results['per_class_metrics'].append(per_class_metrics)
        
        print(f"\n📊 第 {fold} 折结果 (23分类):")
        print(f"   - 最佳验证 F1: {best_f1:.4f}")
        print(f"   - 测试 F1: {test_f1:.4f}")
        print(f"   - 测试 AUC: {test_auc:.4f}")
        print(f"   - 用时: {fold_time/60:.2f} 分钟")
        
        # 保存详细的分类报告
        with open(fold_output_dir / 'classification_report_23classes.txt', 'w') as f:
            f.write(f"Fold {fold} Classification Report (23 Classes)\n")
            f.write("="*60 + "\n")
            f.write(classification_report(test_true, test_preds, target_names=classes, zero_division=0))
            f.write(f"\nTest F1-Score: {test_f1:.4f}\n")
            f.write(f"Test AUC: {test_auc:.4f}\n")
            
            # 添加每个类别的详细指标
            f.write(f"\nPer-Class Performance (23 Classes):\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}\n")
            f.write("-"*65 + "\n")
            for class_name, metrics in per_class_metrics.items():
                f.write(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1_score']:<10.4f} {metrics['auc']:<10.4f}\n")
        
        # 🔥 每完成一个fold就保存一次结果，防止再次中断
        if len(all_fold_results['test_f1']) > 0:
            temp_results = {
                'model_version': 'MeDeA_23Classes',
                'total_classes': len(classes),
                'completed_folds': len(all_fold_results['test_f1']),
                'cross_validation_results': all_fold_results,
                'summary_statistics': {
                    'mean_test_f1': np.mean(all_fold_results['test_f1']),
                    'std_test_f1': np.std(all_fold_results['test_f1']),
                    'mean_test_auc': np.mean(all_fold_results['test_auc']),
                    'std_test_auc': np.std(all_fold_results['test_auc']),
                    'total_time_minutes': sum(all_fold_results['fold_times'])/60
                },
            }
            
            with open(output_dir / 'cross_validation_summary_23classes.json', 'w') as f:
                json.dump(temp_results, f, indent=2)
            
            print(f"💾 已保存前 {len(all_fold_results['test_f1'])} 折的临时结果")
    
    # 计算并保存详细的交叉验证总结果
    if all_fold_results['test_f1'] and classes is not None:
        print(f"\n{'='*60}")
        print("🎉 23分类十折交叉验证完成！")
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
        
        print(f"📈 23分类交叉验证结果总结:")
        print(f"   - 类别总数: {len(classes)}")
        print(f"   - 验证 F1-Score: {mean_val_f1:.4f} ± {std_val_f1:.4f}")
        print(f"   - 测试 F1-Score: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"   - 测试 AUC:      {mean_test_auc:.4f} ± {std_test_auc:.4f}")
        print(f"   - 总用时: {total_time/60:.2f} 分钟")
        
        # 打印每个类别的平均性能（前10个类别）
        print(f"\n📊 Per-Class Performance Summary (前10个类别):")
        print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'AUC':<15}")
        print("-" * 80)
        for i, (class_name, metrics) in enumerate(per_class_summary.items()):
            if i >= 10:  # 只显示前10个类别
                break
            precision_str = f"{metrics['precision_mean']:.3f}±{metrics['precision_std']:.3f}"
            recall_str = f"{metrics['recall_mean']:.3f}±{metrics['recall_std']:.3f}"
            f1_str = f"{metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f}"
            auc_str = f"{metrics['auc_mean']:.3f}±{metrics['auc_std']:.3f}"
            
            print(f"{class_name:<15} {precision_str:<15} {recall_str:<15} {f1_str:<15} {auc_str:<15}")
        
        if len(classes) > 10:
            print(f"... 以及其他 {len(classes)-10} 个类别的结果")
        
        # 保存交叉验证结果总结
        results_summary = {
            'model_version': 'MeDeA_23Classes',
            'total_classes': len(classes),
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
        with open(output_dir / 'cross_validation_summary_23classes.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 保存详细的文本报告
        with open(output_dir / 'cross_validation_report_23classes.txt', 'w') as f:
            f.write("MeDeA Model - 23 Classes 10-Fold Cross Validation Results\n")
            f.write("="*80 + "\n\n")
            
            # 总体性能统计
            f.write("Overall Performance Summary:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Classes: {len(classes)}\n")
            f.write(f"Validation F1-Score: {mean_val_f1:.4f} ± {std_val_f1:.4f}\n")
            f.write(f"Test F1-Score:       {mean_test_f1:.4f} ± {std_test_f1:.4f}\n")
            f.write(f"Test AUC:            {mean_test_auc:.4f} ± {std_test_auc:.4f}\n")
            f.write(f"Total Time:          {total_time/60:.2f} minutes\n\n")
            
            # Per-Class Performance Table
            f.write("TABLE - PER-CLASS PERFORMANCE OF MEDEA (23 CLASSES).\n")
            f.write("="*80 + "\n")
            f.write(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'AUC':<15}\n")
            f.write("-"*80 + "\n")
            
            for class_name, metrics in per_class_summary.items():
                precision_str = f"{metrics['precision_mean']*100:.2f}%"
                recall_str = f"{metrics['recall_mean']*100:.2f}%"
                f1_str = f"{metrics['f1_mean']*100:.2f}%"
                auc_str = f"{metrics['auc_mean']*100:.2f}%"
                
                f.write(f"{class_name:<15} {precision_str:<15} {recall_str:<15} {f1_str:<15} {auc_str:<15}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            
            # Individual Fold Results
            f.write("Individual Fold Results:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Fold':<6} {'Val F1':<8} {'Test F1':<8} {'Test AUC':<8} {'Time(min)':<10}\n")
            f.write("-"*80 + "\n")
            for i, (vf1, tf1, tauc, t) in enumerate(zip(
                all_fold_results['val_f1'], all_fold_results['test_f1'], 
                all_fold_results['test_auc'], all_fold_results['fold_times']
            ), 1):
                f.write(f"Fold {i:<2} {vf1:<8.4f} {tf1:<8.4f} {tauc:<8.4f} {t/60:<10.1f}\n")
        
        print(f"📁 所有结果已保存到: {output_dir}")
        print(f"📊 23分类详细性能报告已保存到: {output_dir}/cross_validation_report_23classes.txt")
    else:
        print("❌ 没有成功完成任何fold的训练")

# ==============================================================================
# 6. 主程序入口 (Main Entry)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🚀 MeDeA 23分类模型训练脚本')
    
    # 🔥 修改7: 更新数据路径和输出路径
    parser.add_argument('--data_dir', type=str, 
                       default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data_23subclasses',
                       help='包含23分类十折交叉验证数据的目录路径')
    parser.add_argument('--output_dir', type=str, 
                       default='./saved_models/medea_23classes_experiment', 
                       help='保存模型和结果的目录')
    
    # 🔥 新增：继续实验的参数
    parser.add_argument('--start_fold', type=int, default=1, 
                       help='从第几折开始进行交叉验证 (1-10)')
    
    parser.add_argument('--epochs', type=int, default=120)  # 🔥 增加训练轮数
    parser.add_argument('--patience', type=int, default=20)  # 🔥 增加早停耐心值
    parser.add_argument('--batch_size', type=int, default=512)  # 🔥 减少批次大小以适应更大模型
    parser.add_argument('--lr', type=float, default=8e-4)  # 🔥 调整学习率
    parser.add_argument('--weight_decay', type=float, default=1.5e-4)  # 🔥 增加权重衰减
    
    # 🔥 修改8: 调整模型超参数以适应23分类
    parser.add_argument('--d_model', type=int, default=512, help='注意力头的隐藏维度 (增大以处理23分类)')
    parser.add_argument('--base_filters', type=int, default=64, help='CNN骨干网络的基础滤波器数量')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比率 (降低以避免欠拟合)')
    
    parser.add_argument('--verbose', action='store_true', help='显示详细的训练过程')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    
    args = parser.parse_args()
    
    print("🚀 启动 MeDeA 23分类模型十折交叉验证...")
    run_cross_validation_23classes(args)