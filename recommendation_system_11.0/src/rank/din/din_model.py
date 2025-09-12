# src/rank/din/din_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """DIN 注意力层"""
    
    def __init__(self, embed_dim, hidden_dim=64):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, query, keys, values, mask=None):
        """
        Args:
            query: [batch_size, embed_dim] - 候选物品embedding
            keys: [batch_size, seq_len, embed_dim] - 历史行为序列
            values: [batch_size, seq_len, embed_dim] - 历史行为序列 (通常与keys相同)
            mask: [batch_size, seq_len] - 序列mask，1表示有效位置
        """
        batch_size, seq_len, embed_dim = keys.size()
        
        # 扩展query维度用于计算注意力
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embed_dim]
        
        # 计算注意力输入特征
        # 拼接: query, key, query-key, query*key
        query_key_concat = torch.cat([
            query_expanded,  # query
            keys,            # key
            query_expanded - keys,  # query - key
            query_expanded * keys   # query * key
        ], dim=-1)  # [batch_size, seq_len, embed_dim * 4]
        
        # 计算注意力分数
        attention_scores = self.attention_net(query_key_concat).squeeze(-1)  # [batch_size, seq_len]
        
        # 应用mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # 加权聚合
        output = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)  # [batch_size, embed_dim]
        
        return output, attention_weights


class DINModel(nn.Module):
    """DIN 模型"""
    
    def __init__(self, 
                 user_cate_dims,      # 用户类别特征维度
                 user_numeric_dim,    # 用户数值特征维度
                 embed_dim=64,        # embedding维度
                 max_seq_len=50,      # 最大序列长度
                 hidden_dims=[128, 64]):  # MLP隐藏层维度
        super(DINModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 用户特征embedding
        self.user_embeddings = nn.ModuleDict()
        for feat_name, feat_dim in user_cate_dims.items():
            self.user_embeddings[feat_name] = nn.Embedding(feat_dim, embed_dim)
        
        # 物品ID embedding（简化版本）
        self.item_embedding = nn.Embedding(100000, embed_dim)  # 假设最多100000个物品
        
        # 历史行为序列embedding（使用物品ID）
        self.seq_embedding = nn.Embedding(100000, embed_dim)
        
        # 注意力层
        self.attention_layer = AttentionLayer(embed_dim)
        
        # 数值特征处理
        self.user_numeric_proj = nn.Linear(user_numeric_dim, embed_dim) if user_numeric_dim > 0 else None
        
        # 召回分数特征
        self.recall_scores_proj = nn.Linear(3, embed_dim)  # CF, keyword, youtube_dnn 三个分数
        
        # 计算MLP输入维度
        mlp_input_dim = (
            len(user_cate_dims) * embed_dim +  # 用户类别特征
            (embed_dim if user_numeric_dim > 0 else 0) +  # 用户数值特征
            embed_dim +  # 物品ID特征
            embed_dim +  # 注意力聚合后的序列特征
            embed_dim    # 召回分数特征
        )
        
        # MLP层
        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, 
                user_cate_feats,      # 用户类别特征
                user_numeric_feats,   # 用户数值特征
                item_ids,             # 物品ID
                seq_cate_feats,       # 历史行为序列类别特征
                recall_scores,        # 召回分数 [batch_size, 3]
                seq_mask=None):       # 序列mask
        """
        Args:
            user_cate_feats: dict of [batch_size] tensors
            user_numeric_feats: [batch_size, user_numeric_dim] or None
            item_ids: [batch_size] - 物品ID
            seq_cate_feats: dict of [batch_size, seq_len] tensors
            recall_scores: [batch_size, 3] - CF, keyword, youtube_dnn scores
            seq_mask: [batch_size, seq_len] or None
        """
        batch_size = recall_scores.size(0)
        
        # 用户特征embedding
        user_embeds = []
        for feat_name, feat_values in user_cate_feats.items():
            embed = self.user_embeddings[feat_name](feat_values)
            user_embeds.append(embed)
        
        # 用户数值特征
        if self.user_numeric_proj is not None and user_numeric_feats is not None:
            user_numeric_embed = self.user_numeric_proj(user_numeric_feats)
            user_embeds.append(user_numeric_embed)
        
        # 物品特征embedding（简化版本）
        item_embed = self.item_embedding(item_ids)  # [batch_size, embed_dim]
        
        # 历史行为序列embedding（简化版本）
        if seq_cate_feats:
            # 这里我们需要从序列中获取物品ID，但当前实现中没有
            # 暂时使用零向量
            seq_embed = torch.zeros(batch_size, self.max_seq_len, self.embed_dim, 
                                  device=recall_scores.device)
        else:
            seq_embed = torch.zeros(batch_size, self.max_seq_len, self.embed_dim, 
                                  device=recall_scores.device)
        
        # 候选物品特征（用于注意力计算）
        candidate_embed = item_embed  # [batch_size, embed_dim]
        
        # 注意力机制
        if seq_embed.size(1) > 0:  # 如果有序列数据
            attended_seq, attention_weights = self.attention_layer(
                candidate_embed, seq_embed, seq_embed, seq_mask
            )
        else:
            attended_seq = torch.zeros(batch_size, self.embed_dim, device=recall_scores.device)
            attention_weights = None
        
        # 召回分数特征
        recall_embed = self.recall_scores_proj(recall_scores)
        
        # 拼接所有特征
        all_embeds = user_embeds + [item_embed, attended_seq, recall_embed]
        mlp_input = torch.cat(all_embeds, dim=-1)
        
        # MLP预测
        output = self.mlp(mlp_input)
        
        return output.squeeze(-1), attention_weights


def create_din_model(config):
    """根据配置创建DIN模型"""
    return DINModel(
        user_cate_dims=config["user_cate_dims"],
        user_numeric_dim=config["user_numeric_dim"],
        item_cate_dims=config["item_cate_dims"],
        item_numeric_dim=config["item_numeric_dim"],
        embed_dim=config["embed_dim"],
        max_seq_len=config["max_seq_len"],
        hidden_dims=config.get("hidden_dims", [128, 64])
    )
