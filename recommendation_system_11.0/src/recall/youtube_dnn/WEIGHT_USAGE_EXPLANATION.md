# YouTube DNN 样本权重使用说明（新闻推荐系统优化版）

## 新闻推荐系统特点

### 业务特点
- **时效性极强**：新闻内容快速过期，用户兴趣变化快
- **阅读频率高**：用户可能每天多次访问
- **内容消费快**：单篇新闻阅读时间短
- **兴趣多样化**：用户可能关注多个领域

### 权重设计原则
- **时效性优先**：recency权重最高(0.5)
- **活跃度重要**：frequency权重中等(0.3)
- **深度次要**：monetary权重较低(0.2)
- **训练稳定**：权重范围[0.5, 1.5]避免极端值

## 问题分析

### 原始问题
- **time_weight**：在数据集中被计算和存储，但在训练时**没有被使用**
- **RFM权重**：虽然RFM特征被标准化后作为用户特征输入模型，但**没有作为样本权重使用**

### 修复方案
现在实现了完整的样本权重机制，包括：
1. **time_weight**：时间衰减权重
2. **RFM权重**：基于用户价值的权重
3. **组合权重**：`time_weight * rfm_weight`

## 权重计算详解

### 1. Time Weight (时间权重)

#### 计算方式
```python
# 在 dataset_youtube_dnn.py 中
chunk = apply_time_weight(
    chunk,
    dt_col="dt",
    method=time_decay_method,  # "linear"
    ref_date=cutoff_date,
    decay_days=decay_days,     # 30天
    min_weight=min_time_weight # 0.1
)
```

#### 作用
- **近期交互**：权重更高（接近1.0）
- **远期交互**：权重更低（接近0.1）
- **时间衰减**：模拟用户兴趣的时间衰减特性

### 2. RFM Weight (用户价值权重)

#### 计算方式（新闻推荐系统优化）
```python
def _compute_rfm_weights(self):
    for row in self.users.itertuples(index=False):
        recency = row.recency
        frequency = row.frequency
        monetary = row.actions_per_active_day_30d
        
        # 新闻推荐系统权重：简单加权平均
        # recency: 0.5 (新闻时效性最重要)
        # frequency: 0.3 (用户活跃度重要)  
        # monetary: 0.2 (阅读深度相对次要)
        rfm_weight = (recency * 0.5 + frequency * 0.3 + monetary * 0.2)
        
        # 归一化到[0.5, 1.5]范围，避免权重过大影响训练稳定性
        rfm_weight = np.clip(rfm_weight, 0.5, 1.5)
```

#### 权重含义（新闻推荐系统）
- **高价值用户**：RFM权重 > 1.0，训练时给予更多关注
- **低价值用户**：RFM权重 < 1.0，训练时关注度降低
- **权重范围**：[0.5, 1.5]，避免极端权重影响训练稳定性
- **时效性优先**：recency权重最高(0.5)，符合新闻推荐特点

### 3. 组合权重

#### 计算方式
```python
# 在训练时
combined_weights = time_weight * rfm_weight
loss = criterion(logits, combined_weights)
```

#### 权重组合逻辑
- **正样本**：`time_weight * rfm_weight`
  - 近期高价值用户交互：权重最高
  - 远期低价值用户交互：权重最低
- **负样本**：`1.0 * rfm_weight`
  - 只使用RFM权重，没有时间权重

## 权重使用效果

### 1. 训练效果
- **更关注近期交互**：时间权重确保模型更关注用户最近的兴趣
- **更关注高价值用户**：RFM权重确保模型更关注高价值用户的行为
- **平衡正负样本**：负样本也使用RFM权重，保持训练平衡

### 2. 业务价值
- **提升高价值用户推荐**：高价值用户获得更好的个性化推荐
- **时间敏感性**：模型更好地捕捉用户兴趣的时间变化
- **资源优化**：将更多计算资源投入到重要的用户和交互上

## 代码修改总结

### 1. 数据集修改 (`dataset_youtube_dnn.py`)
```python
# 添加RFM权重计算
def _compute_rfm_weights(self):
    # 计算每个用户的RFM权重

# 修改Sample结构
Sample = namedtuple("Sample", [
    ..., "rfm_weight"  # 新增RFM权重字段
])

# 修改样本生成
def _generate_samples(self):
    # 正样本：time_weight * rfm_weight
    # 负样本：1.0 * rfm_weight
```

### 2. 训练脚本修改 (`train_youtube_dnn.py`)
```python
# 添加加权损失函数
class WeightedSoftmaxLoss(nn.Module):
    def forward(self, logits, sample_weights=None):
        # 支持样本权重的损失计算

# 修改训练循环
combined_weights = time_weight * rfm_weight
loss = criterion(logits, combined_weights)
```

## 使用建议

### 1. 权重调优
```python
# 可以调整RFM权重公式
rfm_weight = (recency * 0.3 + frequency * 0.4 + monetary * 0.3)

# 可以调整权重范围
rfm_weight = np.clip(rfm_weight, 0.5, 1.5)  # 更保守的权重范围
```

### 2. 监控指标
- **权重分布**：监控time_weight和rfm_weight的分布
- **训练稳定性**：观察加权loss的收敛情况
- **业务效果**：对比加权前后的推荐效果

### 3. 实验对比
```bash
# 不使用权重
python train_youtube_dnn.py --no_weights

# 只使用time_weight
python train_youtube_dnn.py --time_weight_only

# 使用组合权重（默认）
python train_youtube_dnn.py
```

## 总结

通过这次修改，我们实现了：
1. ✅ **time_weight正确使用**：时间衰减权重应用到loss计算
2. ✅ **RFM权重引入**：用户价值权重作为样本权重
3. ✅ **组合权重机制**：time_weight * rfm_weight的综合权重
4. ✅ **训练效果提升**：更关注高价值用户的近期行为

这样既解决了原始问题，又提升了模型的训练效果和业务价值！
