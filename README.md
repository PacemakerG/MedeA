# ECG分类任务综合实验项目

## 🎯 项目概述

本项目对比了多种深度学习模型在PTB-XL ECG数据集上的分类性能，包括传统基线模型和创新的CC-CNN-Mamba混合架构。实验旨在找到最适合ECG分类任务的模型架构。

## 📊 实验结果总结

### 🏆 模型性能排名 (Macro F1)

| 排名 | 模型名称 | Macro F1 | 模型类型 | 特点 |
|------|----------|----------|----------|------|
| 1 | **Inception Baseline** | **0.7477** | 基线模型 | 多尺度特征提取 |
| 2 | ResNet Baseline | 0.7464 | 基线模型 | 残差连接 |
| 3 | EfficientNet Baseline | 0.7438 | 基线模型 | 深度可分离卷积 |
| 4 | DenseNet Baseline | 0.7357 | 基线模型 | 密集连接 |
| 5 | CNN Baseline | 0.7347 | 基线模型 | 经典卷积网络 |
| 6 | RNN Baseline | 0.7319 | 基线模型 | 循环神经网络 |
| 7 | Vision Transformer | 0.4957 | 基线模型 | 自注意力机制 |
| 8 | CC-CNN-Mamba | 0.0000 | 创新模型 | CNN+Mamba混合 |

### 🔍 关键发现

1. **最佳模型**: Inception Baseline 在Macro F1上表现最佳 (0.7477)
2. **基线模型优势**: 传统CNN架构在ECG分类任务上表现稳定且优秀
3. **类别表现**: NORM类别表现最好，病理类别(CD、HYP、MI)更具挑战性
4. **架构特点**: 多尺度特征提取的Inception架构最适合ECG数据

## 🏗️ 项目结构

```
ECG_Project/
├── models/                          # 模型实现
│   ├── baselinemodels/             # 基线模型
│   │   ├── cnn_baseline.py        # CNN基线模型
│   │   ├── rnn_baseline.py        # RNN (LSTM/GRU)基线模型
│   │   ├── resnet_baseline.py     # ResNet基线模型
│   │   ├── densenet_baseline.py   # DenseNet基线模型
│   │   ├── inception_baseline.py  # Inception基线模型
│   │   ├── efficientnet_baseline.py # EfficientNet基线模型
│   │   └── vit_baseline.py        # Vision Transformer基线模型
│   ├── ptbxl_cnn_cc_biomamba.py   # 原始CC-CNN-Mamba模型
│   ├── ptbxl_cnn_cc_biomamba_v2.py # V2改进版本
│   ├── ptbxl_cnn_cc_biomamba_v3.py # V3改进版本
│   └── ptbxl_cnn_cc_biomamba_final.py # 最终优化版本
├── processed_data/                 # 预处理数据
├── saved_models/                   # 保存的模型权重
├── results/                        # 实验结果
├── experiment_visualizations/      # 可视化图表
├── run_comprehensive_experiment.py # 综合实验运行脚本
├── generate_comprehensive_report.py # 实验报告生成脚本
└── README.md                       # 项目说明文档
```

## 🚀 快速开始

### 环境要求

```bash
# 激活conda环境
conda activate LLM

# 设置临时目录
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
```

### 运行单个模型

```bash
# 运行CNN基线模型
python models/baselinemodels/cnn_baseline.py --epochs 20 --batch_size 32 --lr 0.001

# 运行RNN基线模型 (LSTM)
python models/baselinemodels/rnn_baseline.py --epochs 20 --batch_size 32 --lr 0.001 --rnn_type lstm

# 运行CC-CNN-Mamba模型
python models/ptbxl_cnn_cc_biomamba.py --epochs 20 --batch_size 32 --lr 0.001
```

### 运行综合实验

```bash
# 运行所有模型的综合实验
python models/run_comprehensive_experiment.py
```

### 生成实验报告

```bash
# 生成综合实验报告和可视化
python models/generate_comprehensive_report.py
```

## 📈 实验设置

### 数据集
- **数据集**: PTB-XL ECG数据集
- **采样率**: 100Hz
- **数据分割**: 训练集(17,441)、验证集(2,193)、测试集(2,203)
- **类别**: 5个ECG类别 (CD, HYP, MI, NORM, STTC)

### 训练参数
- **训练轮数**: 20 epochs
- **批次大小**: 32
- **学习率**: 0.001
- **优化器**: AdamW
- **损失函数**: BCEWithLogitsLoss
- **评估指标**: Macro F1, Micro F1, Weighted F1

## 🎨 可视化结果

项目生成了两种主要的可视化图表：

1. **总体性能对比图** (`overall_performance_comparison.png`)
   - Macro F1、Micro F1、Weighted F1对比
   - 最佳模型性能雷达图

2. **类别性能热力图** (`class_performance_heatmap.png`)
   - 各类别F1分数对比
   - 模型间性能差异分析

## 📋 详细实验报告

完整的实验报告请查看：
- **Markdown格式**: `ECG_Classification_Experiment_Report.md`
- **可视化图表**: `experiment_visualizations/` 目录

## 🔬 技术特点

### 基线模型
- **CNN**: 经典卷积网络，适合空间特征提取
- **RNN**: LSTM/GRU，适合时序建模
- **ResNet**: 残差连接，解决深度网络训练问题
- **DenseNet**: 密集连接，特征重用
- **Inception**: 多尺度特征提取
- **EfficientNet**: 深度可分离卷积，计算效率高
- **Vision Transformer**: 自注意力机制，长序列建模

### 创新模型
- **CC-CNN-Mamba**: 结合CNN的空间建模能力和Mamba的序列建模能力
- **混合架构**: 多分支特征融合，注意力机制

## 📊 性能分析

### 模型优势分析
1. **Inception Baseline**: 多尺度特征提取最适合ECG数据
2. **ResNet Baseline**: 残差连接提供稳定的训练过程
3. **EfficientNet Baseline**: 高效的网络架构设计

### 改进空间
1. **数据增强**: 改善类别不平衡问题
2. **超参数优化**: 进一步调优学习率和网络结构
3. **集成学习**: 结合多个模型的优势

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证。

## 📞 联系方式

如有问题或建议，请通过Issue联系。

---

*最后更新: 2025-08-17*
*实验完成状态: ✅ 已完成*
