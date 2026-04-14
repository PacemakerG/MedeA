# -*- coding: utf-8 -*-
"""
综合ECG分类实验脚本

运行所有基线模型和CC-CNN-Mamba模型，设置相同的epochs进行公平对比
"""

import os
import subprocess
import time
import argparse
from datetime import datetime

def run_model_training(model_name, script_path, args, timeout_seconds):
    """运行单个模型训练"""
    print(f"\n{'='*80}")
    print(f"🚀 开始训练 {model_name}")
    print(f"⏰ 超时设置为: {timeout_seconds / 3600:.1f} 小时 ({timeout_seconds}秒)")
    print(f"{'='*80}")
    
    # 构建命令
    cmd = ["python3", script_path] + args
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行训练
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} 训练成功完成!")
            print(f"训练耗时: {duration/60:.1f} 分钟")
            return True, duration
        else:
            print(f"❌ {model_name} 训练失败!")
            print(f"错误输出: {result.stderr}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} 训练超时 ({timeout_seconds / 3600:.1f}小时)")
        return False, timeout_seconds
    except Exception as e:
        print(f"❌ {model_name} 训练异常: {e}")
        return False, 0

def find_data_file(default_path):
    """查找数据文件"""
    if os.path.exists(default_path):
        return default_path
    
    print(f"❌ 默认数据文件不存在: {default_path}")
    print("🔍 正在搜索可能的数据文件位置...")
    
    possible_paths = [
        "/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data/ptbxl_processed_100hz.npz",
        "/home/elonge/WorkSpace/ECG_Project/processed_data/ptbxl_processed_100hz.npz",
        "./processed_data/ptbxl_processed_100hz.npz",
        "../processed_data/ptbxl_processed_100hz.npz",
        "../../processed_data/ptbxl_processed_100hz.npz"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到数据文件: {path}")
            return path
        else:
            print(f"❌ 不存在: {path}")
    
    print("❌ 无法找到数据文件，请检查路径或重新预处理数据")
    return None

def main():
    """主函数"""
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='🚀 综合ECG分类实验脚本')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=30, help='统一的训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='统一的批量大小')  
    parser.add_argument('--lr', type=float, default=1e-3, help='统一的学习率')
    
    # 数据和路径参数
    parser.add_argument('--data_file', type=str, 
                        default='/home/elonge/WorkSpace/ECG_Project/PTXBL-ECG/processed_data/ptbxl_processed_100hz.npz',
                        help='数据文件路径')
    
    # 实验控制参数
    parser.add_argument('--models', nargs='*', help='指定要训练的模型列表（可选，不指定则训练所有模型）')
    parser.add_argument('--wait_time', type=int, default=30, help='模型训练间隔等待时间（秒）')
    
    # 【修改点】: 将默认超时从 7200 (2小时) 改为 25200 (7小时)
    parser.add_argument('--timeout', type=int, default=25200, help='单个模型训练超时时间（秒）')
    
    args = parser.parse_args()
    
    print("🚀 综合ECG分类实验 - 基线模型 vs CC-CNN-Mamba模型")
    print("=" * 80)
    
    # 调试信息：显示解析的参数
    print(f"🔧 解析的参数:")
    print(f"   - epochs: {args.epochs}")
    print(f"   - batch_size: {args.batch_size}")
    print(f"   - lr: {args.lr}")
    print(f"   - models: {args.models}")
    print(f"   - data_file: {args.data_file}")
    print(f"   - timeout: {args.timeout} 秒 ({args.timeout / 3600:.1f} 小时) [默认值]")

    
    # 设置环境变量
    os.environ['TMPDIR'] = os.path.expanduser('~/tmp')
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    
    # 创建必要的目录
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # 使用命令行参数
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    
    # 查找数据文件
    DATA_FILE = find_data_file(args.data_file)
    if DATA_FILE is None:
        return
    
    print(f"✅ 使用数据文件: {DATA_FILE}")
    
    # 所有模型配置 - 添加数据文件路径
    all_models_config = {
        # 基线模型
        "CNN_Baseline": {
            "script": "models/baselinemodels/cnn_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE), 
                     "--data_file", DATA_FILE]
        },
        "RNN_LSTM_Baseline": {
            "script": "models/baselinemodels/rnn_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE), 
                     "--rnn_type", "lstm", "--data_file", DATA_FILE]
        },
        "RNN_GRU_Baseline": {
            "script": "models/baselinemodels/rnn_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE), 
                     "--rnn_type", "gru", "--data_file", DATA_FILE]
        },
        "ResNet_Baseline": {
            "script": "models/baselinemodels/resnet_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE),
                     "--data_file", DATA_FILE]
        },
        "DenseNet_Baseline": {
            "script": "models/baselinemodels/densenet_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE),
                     "--data_file", DATA_FILE]
        },
        "Inception_Baseline": {
            "script": "models/baselinemodels/inception_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE),
                     "--data_file", DATA_FILE]
        },
        "EfficientNet_Baseline": {
            "script": "models/baselinemodels/efficientnet_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE),
                     "--data_file", DATA_FILE]
        },
        "VisionTransformer_Baseline": {
            "script": "models/baselinemodels/vit_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE), 
                     "--d_model", "128", "--data_file", DATA_FILE]
        },
        "Transformer_Baseline": {
            "script": "models/baselinemodels/transformer_baseline.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE), 
                     "--d_model", "128", "--nhead", "8", "--num_layers", "6", "--data_file", DATA_FILE]
        },
        
        # CC-CNN-Mamba模型系列
        "v2": {
            "script": "models/v9.py",
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE),
                     "--data_file", DATA_FILE]
        },
        "sota": {
            "script": "models/v10sota.py", 
            "args": ["--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE), "--lr", str(LEARNING_RATE),
                     "--data_file", DATA_FILE]
        }
    }
    
    # 如果指定了特定模型，只运行指定的模型
    if args.models:
        models_config = {}
        print(f"\n🎯 用户指定了模型: {args.models}")
        for model_name in args.models:
            if model_name in all_models_config:
                models_config[model_name] = all_models_config[model_name]
                print(f"✅ 找到模型: {model_name}")
            else:
                print(f"⚠️ 未知模型: {model_name}")
                print(f"可用模型: {list(all_models_config.keys())}")
        
        if not models_config:
            print("❌ 没有找到有效的模型进行训练")
            return
    else:
        print(f"\n📋 没有指定特定模型，将训练所有模型")
        models_config = all_models_config
    
    # 训练结果记录
    training_results = {}
    total_start_time = time.time()
    
    print(f"\n📋 计划训练 {len(models_config)} 个模型")
    print(f"统一训练参数: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")
    if args.models:
        print(f"选定模型: {list(models_config.keys())}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 依次训练每个模型
    for i, (model_name, config) in enumerate(models_config.items(), 1):
        print(f"\n📝 准备训练: {model_name} ({i}/{len(models_config)})")
        print(f"脚本路径: {config['script']}")
        print(f"参数: {' '.join(config['args'])}")
        
        # 检查脚本是否存在
        if not os.path.exists(config['script']):
            print(f"❌ 脚本文件不存在: {config['script']}")
            training_results[model_name] = {
                "status": "failed",
                "error": "Script file not found",
                "duration": 0
            }
            continue
        
        # 运行训练
        success, duration = run_model_training(
            model_name, 
            config['script'], 
            config['args'],
            args.timeout
        )
        
        # 记录结果
        training_results[model_name] = {
            "status": "success" if success else "failed",
            "duration": duration,
            "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 等待一下再开始下一个模型（不是最后一个时才等待）
        if success and i < len(models_config):
            print(f"⏳ 等待{args.wait_time}秒后开始下一个模型...")
            time.sleep(args.wait_time)
    
    # 计算总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 输出训练总结
    print(f"\n{'='*80}")
    print(f"🎯 所有模型训练完成!")
    print(f"{'='*80}")
    
    print(f"总耗时: {total_duration/3600:.1f} 小时")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 统计成功和失败的模型
    successful_models = [name for name, result in training_results.items() if result["status"] == "success"]
    failed_models = [name for name, result in training_results.items() if result["status"] == "failed"]
    
    print(f"\n📊 训练结果统计:")
    print(f"   ✅ 成功: {len(successful_models)}/{len(models_config)}")
    print(f"   ❌ 失败: {len(failed_models)}/{len(models_config)}")
    
    if successful_models:
        print(f"\n✅ 成功训练的模型:")
        for model_name in successful_models:
            result = training_results[model_name]
            print(f"   - {model_name}: {result['duration']/60:.1f} 分钟")
    
    if failed_models:
        print(f"\n❌ 训练失败的模型:")
        for model_name in failed_models:
            result = training_results[model_name]
            print(f"   - {model_name}: {result.get('error', 'Unknown error')}")
    
    # 下一步建议
    print(f"\n🎯 下一步建议:")
    print(f"   1. 检查所有模型的结果文件: ls -la ./results/")
    print(f"   2. 运行性能对比分析脚本")
    print(f"   3. 生成综合实验报告")
    print(f"   4. 创建README文档记录实验结果")
    
    print(f"\n✅ 综合实验训练任务完成!")

if __name__ == "__main__":
    main()