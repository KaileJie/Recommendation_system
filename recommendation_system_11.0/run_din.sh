#!/bin/bash

# DIN Model Training and Inference Script

echo "🚀 Starting DIN Model Pipeline..."

# 设置环境变量
export PYTHONPATH=/Users/dallylovely/Desktop/CCGG/Projects/recommendation_system:$PYTHONPATH

# 检查必要文件是否存在
echo "📋 Checking required files..."

# 检查召回结果
if [ ! -f "output/fusion/fusion_recall.csv" ]; then
    echo "❌ Fusion recall results not found. Please run fusion recall first."
    exit 1
fi

# 检查数据文件
if [ ! -f "data/processed/user_profile_feature.csv" ]; then
    echo "❌ User profile data not found."
    exit 1
fi

if [ ! -f "data/raw/item_metadata.csv" ]; then
    echo "❌ Item metadata not found."
    exit 1
fi

if [ ! -f "data/raw/user_behavior_log_info.csv" ]; then
    echo "❌ Behavior log not found."
    exit 1
fi

echo "✅ All required files found."

# 创建输出目录
mkdir -p output/din

# 1. 训练DIN模型
echo "🎯 Step 1: Training DIN Model..."
python src/rank/din/train_din.py

if [ $? -ne 0 ]; then
    echo "❌ DIN training failed."
    exit 1
fi

echo "✅ DIN model training completed."

# 2. 运行DIN推理
echo "🎯 Step 2: Running DIN Inference..."
python src/rank/din/infer_din.py

if [ $? -ne 0 ]; then
    echo "❌ DIN inference failed."
    exit 1
fi

echo "✅ DIN inference completed."

# 3. 评估DIN模型
echo "🎯 Step 3: Evaluating DIN Model..."
python src/rank/din/evaluate_din.py

if [ $? -ne 0 ]; then
    echo "❌ DIN evaluation failed."
    exit 1
fi

echo "✅ DIN evaluation completed."

# 4. 显示结果摘要
echo "📊 DIN Model Results Summary:"
echo "================================"

if [ -f "output/din/din_eval_metrics.csv" ]; then
    echo "📈 Evaluation Metrics:"
    cat output/din/din_eval_metrics.csv
    echo ""
fi

if [ -f "output/din/din_infer.csv" ]; then
    echo "📦 Inference Results:"
    echo "   Total recommendations: $(wc -l < output/din/din_infer.csv)"
    echo "   Unique users: $(cut -d',' -f1 output/din/din_infer.csv | sort -u | wc -l)"
    echo ""
fi

echo "🎉 DIN Model Pipeline Completed Successfully!"
echo "📁 Results saved in: output/din/"
echo "   - din_model.pt: Trained model"
echo "   - din_infer.csv: Inference results"
echo "   - din_eval_metrics.csv: Evaluation metrics"
echo "   - din_loss_log.csv: Training loss log"
