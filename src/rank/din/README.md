# DIN (Deep Interest Network) 模型

## 概述

DIN (Deep Interest Network) 是一种用于推荐系统的深度学习模型，通过注意力机制动态建模用户兴趣，能够更好地处理用户行为序列和特征交互。

## 模型特点

### 🎯 核心优势
1. **注意力机制**: 根据候选物品动态调整用户历史行为序列的权重
2. **序列建模**: 更好地捕捉用户兴趣的多样性和演化
3. **特征交互**: 学习用户特征、物品特征和行为序列之间的复杂交互
4. **多源融合**: 整合 CF、Keyword、YouTube DNN 三种召回结果

### 🏗️ 架构组件
1. **Embedding Layer**: 将稀疏特征转换为稠密向量
2. **Attention Layer**: 计算候选物品与历史行为的注意力权重
3. **Pooling Layer**: 加权聚合历史行为序列
4. **MLP Layers**: 多层感知机进行最终预测

## 文件结构

```
src/rank/din/
├── din_model.py          # DIN 模型定义
├── dataset_din.py        # 数据集处理
├── train_din.py          # 训练脚本
├── infer_din.py          # 推理脚本
├── evaluate_din.py       # 评估脚本
└── README.md            # 说明文档
```

## 使用方法

### 1. 快速开始

```bash
# 运行完整的 DIN 流程
./run_din.sh
```

### 2. 分步执行

#### 训练模型
```bash
export PYTHONPATH=/path/to/project:$PYTHONPATH
python src/rank/din/train_din.py
```

#### 运行推理
```bash
python src/rank/din/infer_din.py
```

#### 评估模型
```bash
python src/rank/din/evaluate_din.py
```

## 配置参数

### 模型参数
- `embed_dim`: 嵌入维度 (默认: 16)
- `max_seq_len`: 最大序列长度 (默认: 10)
- `hidden_dims`: MLP隐藏层维度 (默认: [32, 16])
- `learning_rate`: 学习率 (默认: 0.001)
- `batch_size`: 批次大小 (默认: 64)
- `num_epochs`: 训练轮数 (默认: 3)

### 数据参数
- `cutoff_days`: 标签窗口天数 (默认: 7)
- `train_days`: 训练数据窗口天数 (默认: 30) - 与召回模型一致
- `neg_pos_ratio`: 负样本比例 (默认: 2)
- `max_samples`: 最大样本数 (默认: 10000)
- `max_users`: 最大用户数 (默认: 1000)

## 输入特征

### 用户特征
- **类别特征**: gender, age_range, city, cluster_id
- **数值特征**: recency, frequency, monetary, rfm_score, actions_per_active_day_30d, 时间偏好等

### 物品特征
- **类别特征**: category, brand
- **数值特征**: price, rating, review_count

### 召回分数
- CF 分数
- Keyword 分数
- YouTube DNN 分数

### 用户行为序列
- 历史点击物品序列
- 序列长度: 最多50个物品
- 注意力权重: 根据候选物品动态调整

## 输出结果

### 训练输出
- `din_model.pt`: 训练好的模型权重
- `din_config.json`: 模型配置
- `din_loss_log.csv`: 训练损失日志

### 推理输出
- `din_infer.csv`: 推荐结果
  - user_id: 用户ID
  - item_id: 物品ID
  - rank: 排序位置
  - din_score: DIN模型分数
  - cf_score: CF分数
  - keyword_score: Keyword分数
  - dnn_score: YouTube DNN分数

### 评估输出
- `din_eval_metrics.csv`: 评估指标
  - Recall@20: 召回率
  - Precision@20: 精确率
  - HitRate@20: 命中率
  - NDCG@20: 归一化折扣累积增益

## 与 Meta-Ranker 对比

DIN 模型相比传统的 Meta-Ranker 有以下优势：

1. **序列建模**: 能够利用用户历史行为序列信息
2. **注意力机制**: 动态调整不同历史行为的重要性
3. **深度学习**: 能够学习更复杂的特征交互
4. **端到端训练**: 整个模型可以端到端优化

## 注意事项

1. **数据依赖**: 需要先运行融合召回 (`fusion_recall.py`) 生成输入数据
2. **内存需求**: 序列数据需要较多内存，建议调整 `batch_size`
3. **训练时间**: 深度学习模型训练时间较长，建议使用 GPU
4. **特征工程**: 确保所有特征都已正确预处理和编码

## 故障排除

### 常见问题

1. **内存不足**: 减小 `batch_size` 或 `max_seq_len`
2. **训练不收敛**: 调整学习率或增加训练轮数
3. **特征缺失**: 检查数据预处理是否正确
4. **模型加载失败**: 确保模型文件路径正确

### 性能优化

1. **使用 GPU**: 设置 `CUDA_VISIBLE_DEVICES`
2. **数据并行**: 使用 `DataParallel` 或 `DistributedDataParallel`
3. **混合精度**: 使用 `torch.cuda.amp` 加速训练
4. **数据缓存**: 将预处理数据缓存到磁盘

## 扩展功能

### 自定义特征
可以在 `dataset_din.py` 中添加新的特征类型：

```python
# 添加新的类别特征
self.user_cate_dims["new_feature"] = feature_dim

# 添加新的数值特征
self.user_numeric_cols.append("new_numeric_feature")
```

### 模型改进
可以在 `din_model.py` 中修改模型架构：

```python
# 修改注意力层
self.attention_layer = CustomAttentionLayer(embed_dim)

# 添加新的网络层
self.custom_layer = nn.Linear(embed_dim, embed_dim)
```

## 参考文献

1. Zhou, G., et al. "Deep Interest Network for Click-Through Rate Prediction." KDD 2018.
2. Zhou, G., et al. "Deep Interest Evolution Network for Click-Through Rate Prediction." AAAI 2019.
3. Pi, Q., et al. "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction." CIKM 2020.
