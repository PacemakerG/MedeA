# -*- coding: utf-8 -*-
"""
调试模型结果文件脚本

检查CC-CNN-Mamba模型结果文件的内容，定位性能为0的问题
"""

import os
import numpy as np
import json
from datetime import datetime

def debug_model_file(file_path, model_name):
    """详细调试单个模型文件"""
    print(f"\n{'='*60}")
    print(f"🔍 调试模型: {model_name}")
    print(f"📁 文件路径: {file_path}")
    print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    print(f"📊 文件大小: {file_size} bytes ({file_size/1024:.2f} KB)")
    
    try:
        # 尝试加载文件
        print(f"🔄 正在加载文件...")
        data = np.load(file_path, allow_pickle=True)
        
        # 检查数据类型
        print(f"📋 数据类型: {type(data)}")
        
        # 如果是.item()格式的字典
        if hasattr(data, 'item'):
            try:
                data_dict = data.item()
                print(f"📋 转换后类型: {type(data_dict)}")
                
                if isinstance(data_dict, dict):
                    print(f"🗂️ 字典键数量: {len(data_dict)}")
                    print(f"🔑 所有键: {list(data_dict.keys())}")
                    
                    # 详细检查每个键的内容
                    for key, value in data_dict.items():
                        print(f"\n🔍 键: '{key}'")
                        print(f"   - 类型: {type(value)}")
                        
                        if isinstance(value, np.ndarray):
                            print(f"   - 形状: {value.shape}")
                            print(f"   - 数据类型: {value.dtype}")
                            print(f"   - 范围: [{np.min(value):.4f}, {np.max(value):.4f}]")
                            print(f"   - 均值: {np.mean(value):.4f}")
                            print(f"   - 非零元素: {np.count_nonzero(value)}")
                            
                            # 如果是预测或标签数组，显示前几个值
                            if key in ['test_predictions', 'test_true', 'y_pred', 'y_true']:
                                print(f"   - 前5个值: {value.flat[:5]}")
                                if value.ndim == 2:
                                    print(f"   - 每类预测数量: {np.sum(value, axis=0)}")
                                    
                        elif isinstance(value, (int, float)):
                            print(f"   - 值: {value}")
                        elif isinstance(value, str):
                            print(f"   - 值: '{value}'")
                        elif isinstance(value, (list, tuple)):
                            print(f"   - 长度: {len(value)}")
                            print(f"   - 内容: {value}")
                        else:
                            print(f"   - 值: {value}")
                    
                    # 检查关键性能指标
                    performance_keys = ['macro_f1', 'micro_f1', 'weighted_f1', 'best_val_f1', 
                                      'test_macro_f1', 'test_micro_f1', 'test_weighted_f1',
                                      'ensemble_macro_f1', 'ensemble_micro_f1', 'ensemble_weighted_f1']
                    
                    print(f"\n📊 性能指标检查:")
                    found_metrics = False
                    for metric in performance_keys:
                        if metric in data_dict:
                            found_metrics = True
                            print(f"   ✅ {metric}: {data_dict[metric]}")
                    
                    if not found_metrics:
                        print(f"   ❌ 未找到标准性能指标")
                        
                        # 尝试从预测数据计算性能
                        if 'test_predictions' in data_dict and 'test_true' in data_dict:
                            print(f"\n🔄 尝试从预测数据计算性能...")
                            y_pred = data_dict['test_predictions']
                            y_true = data_dict['test_true']
                            
                            # 导入sklearn计算指标
                            try:
                                from sklearn.metrics import f1_score
                                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                                micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
                                weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                                
                                print(f"   📈 计算得到的性能:")
                                print(f"      - Macro F1: {macro_f1:.4f}")
                                print(f"      - Micro F1: {micro_f1:.4f}")
                                print(f"      - Weighted F1: {weighted_f1:.4f}")
                                
                                # 检查预测分布
                                print(f"   📊 预测分布:")
                                print(f"      - 预测为1的总数: {np.sum(y_pred)}")
                                print(f"      - 真实为1的总数: {np.sum(y_true)}")
                                print(f"      - 预测准确的数量: {np.sum(y_pred == y_true)}")
                                print(f"      - 准确率: {np.sum(y_pred == y_true) / y_true.size:.4f}")
                                
                                return True
                                
                            except ImportError as e:
                                print(f"   ❌ 无法导入sklearn: {e}")
                    
                else:
                    print(f"❌ 数据不是字典格式")
                    
            except Exception as e:
                print(f"❌ 无法转换为字典: {e}")
                
        else:
            print(f"📋 直接数组类型: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"📊 数组形状: {data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return False

def check_results_directory(results_dir="./results"):
    """检查results目录中的所有文件"""
    print(f"🔍 检查结果目录: {os.path.abspath(results_dir)}")
    
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        return
    
    all_files = os.listdir(results_dir)
    npy_files = [f for f in all_files if f.endswith('.npy')]
    
    print(f"📁 目录中共有 {len(all_files)} 个文件，其中 {len(npy_files)} 个.npy文件")
    print(f"📋 所有.npy文件:")
    for i, file in enumerate(npy_files, 1):
        file_path = os.path.join(results_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"   {i:2d}. {file} ({file_size} bytes)")

def main():
    """主函数"""
    print("🚀 CC-CNN-Mamba模型结果调试工具")
    print("=" * 80)
    
    # 1. 检查results目录
    check_results_directory()
    
    # 2. 重点调试CC-CNN-Mamba模型文件
    cc_mamba_files = [
        ("./results/ptbxl_inception_mamba_v9_final.npy", "CC-CNN-Mamba v9 (Final)"),
        ("./results/ptbxl_inception_mamba_v10_sota_results.npy", "CC-CNN-Mamba v10 (SOTA)"),
        ("./results/v2_results.npy", "CC-CNN-Mamba v2"),
        ("./results/sota_results.npy", "CC-CNN-Mamba SOTA")
    ]
    
    successful_debugs = 0
    
    for file_path, model_name in cc_mamba_files:
        success = debug_model_file(file_path, model_name)
        if success:
            successful_debugs += 1
    
    # 3. 对比检查几个成功的基线模型
    print(f"\n{'='*80}")
    print(f"🔍 对比检查：基线模型（作为参考）")
    print(f"{'='*80}")
    
    baseline_files = [
        ("./results/cnn_baseline_results.npy", "CNN Baseline"),
        ("./results/inception_baseline_results.npy", "Inception Baseline")
    ]
    
    for file_path, model_name in baseline_files:
        if os.path.exists(file_path):
            debug_model_file(file_path, model_name)
            break  # 只检查一个作为参考
    
    # 4. 总结
    print(f"\n{'='*80}")
    print(f"🎯 调试总结")
    print(f"{'='*80}")
    
    print(f"✅ 成功调试的CC-CNN-Mamba文件: {successful_debugs}/{len(cc_mamba_files)}")
    
    if successful_debugs == 0:
        print(f"❌ 所有CC-CNN-Mamba模型文件都有问题！")
        print(f"💡 可能的原因:")
        print(f"   1. 训练过程失败，没有保存性能指标")
        print(f"   2. 模型保存格式不一致")
        print(f"   3. 键名不匹配")
        print(f"   4. 训练脚本逻辑错误")
    elif successful_debugs < len(cc_mamba_files):
        print(f"⚠️ 部分CC-CNN-Mamba模型文件有问题")
    else:
        print(f"✅ 所有CC-CNN-Mamba模型文件都正常")
    
    print(f"\n💡 下一步建议:")
    print(f"   1. 根据调试结果修复模型保存逻辑")
    print(f"   2. 重新运行有问题的模型训练")
    print(f"   3. 统一模型结果保存格式")

if __name__ == "__main__":
    main()