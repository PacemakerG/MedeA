# ECG分类任务综合实验报告

## 📋 实验概述

本实验对比了多种深度学习模型在PTB-XL ECG数据集上的分类性能，包括：
- **基线模型**: CNN、RNN (LSTM/GRU)、ResNet、DenseNet、Inception、EfficientNet、Vision Transformer、Transformer
- **创新模型**: CC-CNN-Mamba系列 (CNN + Mamba混合架构的多个版本)

### 发现的模型
总共加载了 **14** 个模型：
- 📊 基线模型: CNN Baseline
- 📊 基线模型: RNN Baseline
- 📊 基线模型: ResNet Baseline
- 📊 基线模型: DenseNet Baseline
- 📊 基线模型: Inception Baseline
- 📊 基线模型: EfficientNet Baseline
- 📊 基线模型: Vision Transformer Baseline
- 📊 基线模型: dataprocesssimple
- 📊 基线模型: loss+simplestruct
- 📊 基线模型: lossimorovedv10
- 📊 基线模型: simplev10sota
- 📊 基线模型: trainimprovedv10sota
- 📊 基线模型: Auto-detected:  Newstructv10
- 📊 基线模型: Auto-detected: Ptbxl Final Optimized Results


### 实验设置
- **数据集**: PTB-XL ECG数据集 (100Hz采样率)
- **评估指标**: Macro F1、Micro F1、Weighted F1

## 🏆 模型性能排名

### 总体性能对比

| 排名 | 模型名称 | Macro F1 | Micro F1 | Weighted F1 | 最佳验证F1 |
|------|----------|----------|----------|-------------|------------|
| 1 | Inception Baseline | 0.7477 | 0.7810 | 0.7773 | 0.7365 |
| 2 | ResNet Baseline | 0.7464 | 0.7767 | 0.7782 | 0.7337 |
| 3 | EfficientNet Baseline | 0.7438 | 0.7795 | 0.7753 | 0.7274 |
| 4 | DenseNet Baseline | 0.7357 | 0.7702 | 0.7688 | 0.7038 |
| 5 | CNN Baseline | 0.7347 | 0.7669 | 0.7670 | 0.7167 |
| 6 | RNN Baseline | 0.7319 | 0.7646 | 0.7642 | 0.7327 |
| 7 | loss+simplestruct | 0.7293 | 0.7618 | 0.7615 | 0.7259 |
| 8 | lossimorovedv10 | 0.7263 | 0.7633 | 0.7599 | 0.7290 |
| 9 | Auto-detected:  Newstructv10 | 0.7109 | 0.7480 | 0.7471 | 0.6998 |
| 10 | simplev10sota | 0.7087 | 0.7471 | 0.7453 | 0.7068 |
| 11 | trainimprovedv10sota | 0.7084 | 0.7450 | 0.7424 | 0.7162 |
| 12 | dataprocesssimple | 0.6983 | 0.7332 | 0.7317 | 0.6985 |
| 13 | Vision Transformer Baseline | 0.4957 | 0.5305 | 0.5462 | 0.1438 |
| 14 | Auto-detected: Ptbxl Final Optimized Results | 0.3964 | 0.4051 | 0.4499 | 0.0000 |


## 📊 详细性能分析

### 1. 最佳模型分析

**🏆 最佳模型: Inception Baseline**

- **Macro F1**: 0.7477
- **Micro F1**: 0.7810  
- **Weighted F1**: 0.7773

### 2. 模型类型性能对比

#### CC-CNN-Mamba模型性能
❌ **未找到CC-CNN-Mamba模型结果**

#### 基线模型性能
找到 **14** 个基线模型：

1. **Inception Baseline** - Macro F1: 0.7477
2. **ResNet Baseline** - Macro F1: 0.7464
3. **EfficientNet Baseline** - Macro F1: 0.7438
4. **DenseNet Baseline** - Macro F1: 0.7357
5. **CNN Baseline** - Macro F1: 0.7347
6. **RNN Baseline** - Macro F1: 0.7319
7. **loss+simplestruct** - Macro F1: 0.7293
8. **lossimorovedv10** - Macro F1: 0.7263
9. **Auto-detected:  Newstructv10** - Macro F1: 0.7109
10. **simplev10sota** - Macro F1: 0.7087
11. **trainimprovedv10sota** - Macro F1: 0.7084
12. **dataprocesssimple** - Macro F1: 0.6983
13. **Vision Transformer Baseline** - Macro F1: 0.4957
14. **Auto-detected: Ptbxl Final Optimized Results** - Macro F1: 0.3964

- **最佳基线模型**: Inception Baseline (Macro F1: 0.7477)
- **基线模型平均Macro F1**: 0.6867


### 3. 类别性能分析

#### 各类别F1分数对比
| 模型 | CD | HYP | MI | NORM | STTC |
|------|-----|-----|-----|-----|-----|
| CNN Baseline | 0.751 | 0.592 | 0.723 | 0.857 | 0.750 |
| RNN Baseline | 0.729 | 0.588 | 0.757 | 0.849 | 0.737 |
| ResNet Baseline | 0.762 | 0.608 | 0.743 | 0.867 | 0.753 |
| DenseNet Baseline | 0.741 | 0.578 | 0.763 | 0.851 | 0.745 |
| Inception Baseline | 0.739 | 0.636 | 0.727 | 0.870 | 0.766 |
| EfficientNet Baseline | 0.731 | 0.608 | 0.740 | 0.864 | 0.775 |
| Vision Transformer Baseline | 0.441 | 0.351 | 0.466 | 0.729 | 0.492 |
| dataprocesssimple | 0.707 | 0.563 | 0.676 | 0.832 | 0.713 |
| loss+simplestruct | 0.740 | 0.595 | 0.713 | 0.856 | 0.742 |
| lossimorovedv10 | 0.741 | 0.586 | 0.712 | 0.858 | 0.734 |
| simplev10sota | 0.715 | 0.547 | 0.709 | 0.846 | 0.727 |
| trainimprovedv10sota | 0.716 | 0.565 | 0.685 | 0.842 | 0.734 |
| Auto-detected:  Newstructv10 | 0.721 | 0.551 | 0.699 | 0.847 | 0.736 |
| Auto-detected: Ptbxl Final Optimized Results | 0.369 | 0.213 | 0.401 | 0.624 | 0.375 |


## 🔍 关键发现

### 1. 性能优势
- **最佳模型**: Inception Baseline 在Macro F1上表现最佳
- **模型类型**: 基线模型 在整体性能上领先

### 2. 类别表现
- **表现最好的类别**: 通常NORM类别表现较好，因为数据相对平衡
- **挑战类别**: CD、HYP、MI等病理类别可能需要更多关注

### 3. 架构特点
- **CNN系列**: 在空间特征提取上表现稳定
- **RNN系列**: 在时序建模上有优势
- **Transformer系列**: 在长序列建模上表现良好
- **CC-CNN-Mamba**: 结合了CNN的空间建模能力和Mamba的序列建模能力

## 📈 改进建议

### 1. 数据层面
- 考虑使用数据增强技术改善类别不平衡
- 探索更多ECG预处理方法

### 2. 模型层面
- 尝试集成学习方法
- 探索更多超参数组合
- 考虑使用预训练模型

### 3. 训练策略
- 使用更长的训练轮数
- 尝试不同的学习率调度策略
- 探索早停策略的优化

## 📁 实验文件

- **模型文件**: 保存在 `./saved_models/` 目录
- **结果文件**: 保存在 `./results/` 目录
- **可视化图表**: 保存在 `./experiment_visualizations/` 目录

## 📅 实验时间

- **报告生成**: 2025-08-25 17:10:11

---

*本报告由ECG分类实验自动生成 - 成功加载14个模型*
