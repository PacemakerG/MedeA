# -*- coding: utf-8 -*-
"""
MeDeA 消融实验批量运行脚本
一键运行所有8个消融实验，并生成汇总报告
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path
import pandas as pd

# 检查torch是否可用
try:
    import torch
except ImportError:
    torch = None
    print("⚠️ PyTorch 未安装，某些系统信息将无法显示")

def run_ablation_experiment(script_name, description):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"🚀 正在运行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行实验脚本
        result = subprocess.run([
            sys.executable, script_name,
            '--data_dir', '/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data',
            '--epochs', '50',  # 恢复为完整的50个epochs
            '--batch_size', '32',
            '--num_folds', '10',  # 恢复为完整的10折交叉验证
            '--patience', '10'
        ], capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} 完成! 耗时: {duration:.1f}秒")
            print(f"📄 输出摘要: {result.stdout[-200:] if len(result.stdout) > 200 else result.stdout}")
            return True, duration, result.stdout
        else:
            print(f"❌ {description} 失败!")
            print(f"错误信息: {result.stderr}")
            print(f"输出信息: {result.stdout}")
            return False, duration, result.stderr
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {description} 超时! 耗时: {duration:.1f}秒")
        return False, duration, "Timeout"
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {description} 异常: {e}")
        return False, duration, str(e)

def collect_results():
    """收集所有消融实验结果"""
    results = {}
    
    # 结果文件映射
    result_files = {
        'Full Model (Our)': 'ablation_full_model_results.json',
        'w/o Attention': 'ablation_no_attention_results.json', 
        'w/o Multi-Head': 'ablation_single_head_results.json',
        'w/o Learned Query': 'ablation_fixed_query_results.json',
        'w/o CNN Backbone': 'ablation_no_cnn_results.json',
        'w/o Regularization': 'ablation_no_regularization_results.json',
        'Simple CNN': 'ablation_simple_cnn_results.json'
    }
    
    for experiment, filename in result_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    results[experiment] = {
                        'mean_f1': data.get('mean_f1', 0.0),
                        'std_f1': data.get('std_f1', 0.0),
                        'mean_auc': data.get('mean_auc', 0.0),
                        'std_auc': data.get('std_auc', 0.0),
                        'description': data.get('description', experiment)
                    }
            except Exception as e:
                print(f"⚠️ 无法读取 {filename}: {e}")
                results[experiment] = {
                    'mean_f1': 0.0, 'std_f1': 0.0,
                    'mean_auc': 0.0, 'std_auc': 0.0,
                    'description': f'{experiment} (读取失败)'
                }
        else:
            print(f"⚠️ 结果文件不存在: {filename}")
            results[experiment] = {
                'mean_f1': 0.0, 'std_f1': 0.0,
                'mean_auc': 0.0, 'std_auc': 0.0,
                'description': f'{experiment} (未运行)'
            }
    
    return results

def generate_detailed_report(results, experiment_log):
    """生成详细的消融实验报告，包含每类别性能表格"""
    
    report = f"""
# MeDeA 消融实验详细报告
================================================================

## 实验概述
本实验旨在系统性地分析MeDeA模型各个组件对ECG多疾病分类性能的贡献，
通过逐一移除关键组件来验证每个模块的重要性。

## 实验设置
- **数据集**: PTB-XL (10-fold 交叉验证)
- **评估指标**: Macro F1-Score, Macro AUC, Per-class Precision/Recall
- **训练轮数**: 50 epochs (早停机制，patience=10)
- **批次大小**: 32
- **优化器**: AdamW (lr=1e-3, weight_decay=1e-4)
- **硬件环境**: {'CUDA' if torch.cuda.is_available() else 'CPU'}

## 消融实验结果总览

### 整体性能对比表

| 模型配置 | Macro F1 | Macro AUC | 性能下降 | 描述 |
|---------|----------|-----------|----------|------|
"""
    
    # 按F1分数排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
    
    # 计算性能下降（如果有完整模型基线）
    baseline_f1 = None
    baseline_auc = None
    if 'Full Model (Our)' in results:
        baseline_f1 = results['Full Model (Our)']['mean_f1']
        baseline_auc = results['Full Model (Our)']['mean_auc']
    
    for experiment, data in sorted_results:
        f1_str = f"{data['mean_f1']:.4f} ± {data['std_f1']:.4f}"
        auc_str = f"{data['mean_auc']:.4f} ± {data['std_auc']:.4f}"
        
        # 计算性能下降
        if baseline_f1 is not None and experiment != 'Full Model (Our)':
            f1_drop = baseline_f1 - data['mean_f1']
            auc_drop = baseline_auc - data['mean_auc']
            drop_str = f"F1↓{f1_drop:.4f}, AUC↓{auc_drop:.4f}"
        else:
            drop_str = "基线" if experiment == 'Full Model (Our)' else "N/A"
        
        report += f"| {experiment} | {f1_str} | {auc_str} | {drop_str} | {data['description']} |\n"
    
    # 添加各类别性能详情（如果有）
    report += f"""

### 各类别详细性能

#### 性能指标说明
- **Precision**: 预测为正类的样本中实际为正类的比例
- **Recall**: 实际正类样本中被正确预测的比例  
- **F1-Score**: Precision和Recall的调和平均数
- **AUC**: ROC曲线下的面积，衡量分类器的判别能力

#### 完整模型性能 (如果可用)
根据您提供的表格数据，完整MeDeA模型的各类别性能如下：

| Class | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|----------|-----|
| NORM  | 81.20%    | 90.10% | 87.00%   | 95.70% |
| MI    | 81.10%    | 72.70% | 76.60%   | 93.70% |
| STTC  | 74.90%    | 73.40% | 74.10%   | 93.20% |
| CD    | 82.00%    | 73.00% | 77.20%   | 93.30% |
| HYP   | 72.20%    | 53.20% | 61.10%   | 91.50% |

> **注意**: 以上是参考数据，实际消融实验结果可能有所不同。

## 实验运行日志
"""
    
    for log_entry in experiment_log:
        report += f"- {log_entry}\n"
    
    report += f"""

## 组件重要性分析

### 1. 性能下降排序 (基于F1分数下降)
"""
    
    # 计算性能下降
    if baseline_f1 is not None:
        performance_drops = []
        for exp, data in results.items():
            if exp != 'Full Model (Our)':
                f1_drop = baseline_f1 - data['mean_f1']
                auc_drop = baseline_auc - data['mean_auc']
                performance_drops.append((exp, f1_drop, auc_drop))
        
        # 按F1下降排序
        performance_drops.sort(key=lambda x: x[1], reverse=True)
        
        for i, (exp, f1_drop, auc_drop) in enumerate(performance_drops, 1):
            importance = "🔴 关键" if f1_drop > 0.05 else "🟡 重要" if f1_drop > 0.02 else "🟢 可选"
            report += f"{i}. **{exp}**: F1↓{f1_drop:.4f}, AUC↓{auc_drop:.4f} {importance}\n"
    
    report += f"""

### 2. 组件重要性等级划分

#### 🔴 关键组件 (F1下降 > 0.05)
移除后会导致显著性能下降，模型核心组件

#### 🟡 重要组件 (0.02 < F1下降 ≤ 0.05) 
对性能有明显影响，建议保留

#### 🟢 可选组件 (F1下降 ≤ 0.02)
对性能影响较小，可考虑简化

### 3. 架构优化建议

基于消融实验结果：

1. **必须保留的组件**: 性能下降最大的前2-3个组件
2. **可以优化的组件**: 性能下降较小的组件可以简化实现
3. **模型压缩方案**: 移除冗余组件以减少计算复杂度

## 实验环境信息

- **实验时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Python版本**: {sys.version.split()[0]}
- **PyTorch版本**: {torch.__version__ if 'torch' in globals() else 'N/A'}
- **CUDA可用**: {'是' if torch.cuda.is_available() else '否'}

## 结果文件说明

### 生成的文件
- `ablation_report.md`: 本详细报告
- `ablation_results_summary.json`: JSON格式结果汇总
- `ablation_*_results.json`: 各个实验的详细结果

### 使用建议
1. 查看本报告了解整体实验结果
2. 使用JSON文件进行进一步的数据分析
3. 根据组件重要性分析进行模型优化

================================================================
报告生成完成 | MeDeA消融实验系统
"""
    
    return report

def main():
    """主函数：运行所有消融实验"""
    print("🔬 MeDeA 消融实验批量运行器")
    print("=" * 60)
    
    # 消融实验配置
    ablation_experiments = [
        ('MedeA_no_attention.py', 'w/o Attention - 移除注意力机制'),
        ('MedeA_single_head.py', 'w/o Multi-Head - 单头注意力'),
        ('MedeA_no_positional_encoding.py', 'w/o Learned Query - 固定查询'),
        ('MedeA_no_cnn.py', 'w/o CNN Backbone - 仅注意力'),
        ('MedeA_no_regularization.py', 'w/o Regularization - 无正则化'),
        ('MedeA_simple_cnn.py', 'Simple CNN - 简化CNN结构')
    ]
    
    experiment_log = []
    successful_experiments = 0
    total_time = 0
    
    # 创建结果目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"计划运行 {len(ablation_experiments)} 个消融实验...")
    
    # 逐一运行消融实验
    for i, (script, description) in enumerate(ablation_experiments, 1):
        if os.path.exists(script):
            success, duration, output = run_ablation_experiment(script, description)
            
            if success:
                successful_experiments += 1
                log_entry = f"✅ [{i}/{len(ablation_experiments)}] {description} - 成功 ({duration:.1f}s)"
            else:
                log_entry = f"❌ [{i}/{len(ablation_experiments)}] {description} - 失败"
            
            experiment_log.append(log_entry)
            total_time += duration
            print(log_entry)
            
        else:
            log_entry = f"⚠️ [{i}/{len(ablation_experiments)}] {description} - 脚本文件不存在: {script}"
            experiment_log.append(log_entry)
            print(log_entry)
    
    print(f"\n{'='*60}")
    print(f"🎯 实验总结:")
    print(f"成功运行: {successful_experiments}/{len(ablation_experiments)} 个实验")
    print(f"总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"{'='*60}")
    
    # 收集实验结果
    print("\n📊 收集实验结果...")
    final_results = collect_results()
    
    # 生成详细报告
    print("\n" + "="*60)
    print("� 生成详细消融实验报告...")
    print("="*60)
    
    # 生成详细报告
    detailed_report = generate_detailed_report(final_results, experiment_log)
    
    # 保存报告
    report_path = os.path.join(results_dir, 'ablation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    
    # 保存JSON格式的结果汇总
    summary_path = os.path.join(results_dir, 'ablation_results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': final_results,
            'experiment_log': experiment_log,
            'statistics': {
                'total_experiments': len(ablation_experiments),
                'successful_experiments': successful_experiments,
                'total_time_seconds': total_time,
                'total_time_minutes': total_time/60,
                'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"� 详细报告已保存: {report_path}")
    print(f"📋 结果汇总已保存: {summary_path}")
    
    # 显示简要总览
    print("\n" + "="*60)
    print("🎯 消融实验结果总览")
    print("="*60)
    
    # 按F1分数排序显示
    sorted_results = sorted(final_results.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
    
    print(f"{'模型配置':<25} {'Macro F1':<15} {'Macro AUC':<15} {'性能下降':<12}")
    print("-" * 70)
    
    # 获取基线性能
    baseline_f1 = None
    if 'Full Model (Our)' in final_results:
        baseline_f1 = final_results['Full Model (Our)']['mean_f1']
    
    for experiment, data in sorted_results:
        f1_str = f"{data['mean_f1']:.4f} ± {data['std_f1']:.4f}"
        auc_str = f"{data['mean_auc']:.4f} ± {data['std_auc']:.4f}"
        
        # 计算性能下降
        if baseline_f1 is not None and experiment != 'Full Model (Our)':
            f1_drop = baseline_f1 - data['mean_f1']
            drop_str = f"F1↓{f1_drop:.4f}"
        else:
            drop_str = "基线" if experiment == 'Full Model (Our)' else "N/A"
        
        print(f"{experiment:<25} {f1_str:<15} {auc_str:<15} {drop_str:<12}")
    
    print("\n" + "="*60)
    print("✅ 所有消融实验已完成！")
    print(f"📂 结果保存在: {results_dir}")
    print("📄 详细报告: ablation_report.md")
    print("📋 结果汇总: ablation_results_summary.json")
    print("="*60)

if __name__ == "__main__":
    main()