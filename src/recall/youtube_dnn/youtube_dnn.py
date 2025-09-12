# src/recall/youtube_dnn/youtube_dnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class YoutubeDNN(nn.Module):
    """
    YouTube DNN: Multi-embedding user tower + numeric features + TF-IDF item features + user history sequence.
    """

    def __init__(self,
                 num_users,
                 num_items,
                 user_cate_dims,     # e.g. {"gender": 3, "age_range": 9, "city": 100, "cluster_id": 20}
                 user_numeric_dim,   # e.g. 3
                 item_text_dim,      # e.g. 300
                 embed_dim=64,
                 hidden_dims=(128, 64),
                 dropout=0.2,
                 max_seq_len=50,     # 用户历史序列最大长度
                 use_sequence=True): # 是否使用历史序列特征
        super(YoutubeDNN, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.use_sequence = use_sequence

        # ---- Embedding layers ----
        # ✅ 添加 padding_idx=0，确保 -1 unknown 编码安全
        self.user_id_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.gender_embedding = nn.Embedding(user_cate_dims["gender"], embed_dim, padding_idx=0)
        self.age_embedding = nn.Embedding(user_cate_dims["age_range"], embed_dim, padding_idx=0)
        self.city_embedding = nn.Embedding(user_cate_dims["city"], embed_dim, padding_idx=0)
        self.cluster_embedding = nn.Embedding(user_cate_dims["cluster_id"], embed_dim, padding_idx=0)

        # ---- User numeric projection ----
        self.user_num_proj = nn.Sequential(
            nn.Linear(user_numeric_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- Item embedding ----
        self.item_id_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)

        # ---- Item text projection ----
        self.item_text_proj = nn.Sequential(
            nn.Linear(item_text_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- User sequence processing ----
        if self.use_sequence:
            # 位置编码
            self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
            
            # 序列pooling层：mean + max + attention
            self.sequence_pooling = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),  # mean + max + attention
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # 简单的attention机制
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1)
            )

        # ---- User tower ----
        # 根据是否使用序列特征调整输入维度
        if self.use_sequence:
            user_input_dim = embed_dim * 7  # 5 embeddings + 1 projected numeric + 1 sequence
        else:
            user_input_dim = embed_dim * 6  # 5 embeddings + 1 projected numeric
        user_layers = []
        for h in hidden_dims:
            user_layers.append(nn.Linear(user_input_dim, h))
            user_layers.append(nn.ReLU())
            user_layers.append(nn.Dropout(dropout))
            user_input_dim = h
        self.user_tower = nn.Sequential(*user_layers)

        # ---- Item tower ----
        item_input_dim = embed_dim * 2  # item_id + text projection
        item_layers = []
        for h in hidden_dims:
            item_layers.append(nn.Linear(item_input_dim, h))
            item_layers.append(nn.ReLU())
            item_layers.append(nn.Dropout(dropout))
            item_input_dim = h
        self.item_tower = nn.Sequential(*item_layers)

        logger.info(f"YoutubeDNN initialized: "
                    f"embed_dim={embed_dim}, hidden_dims={hidden_dims}, dropout={dropout}, "
                    f"user_numeric_dim={user_numeric_dim}, item_text_dim={item_text_dim}, "
                    f"user_cate_dims={user_cate_dims}, use_sequence={use_sequence}, "
                    f"max_seq_len={max_seq_len}")

    def _process_user_sequence(self, item_sequence, sequence_mask):
        """
        处理用户历史序列特征
        
        Args:
            item_sequence: [batch_size, seq_len] item indices
            sequence_mask: [batch_size, seq_len] 序列mask (1表示有效位置)
        
        Returns:
            sequence_embed: [batch_size, embed_dim] 序列特征
        """
        if not self.use_sequence:
            # 如果不使用序列特征，返回零向量
            batch_size = item_sequence.size(0)
            return torch.zeros(batch_size, self.user_id_embedding.embedding_dim, 
                             device=item_sequence.device)
        
        batch_size, seq_len = item_sequence.size()
        
        # 获取item embeddings
        item_embs = self.item_id_embedding(item_sequence)  # [batch_size, seq_len, embed_dim]
        
        # 添加位置编码
        positions = torch.arange(seq_len, device=item_sequence.device).unsqueeze(0).expand(batch_size, -1)
        pos_embs = self.position_embedding(positions)  # [batch_size, seq_len, embed_dim]
        
        # 组合item embedding和位置编码
        sequence_embs = item_embs + pos_embs  # [batch_size, seq_len, embed_dim]
        
        # 应用mask
        sequence_embs = sequence_embs * sequence_mask.unsqueeze(-1).float()
        
        # 多种pooling策略
        # 1. Mean pooling
        valid_lengths = sequence_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
        mean_pooled = sequence_embs.sum(dim=1) / (valid_lengths + 1e-8)  # [batch_size, embed_dim]
        
        # 2. Max pooling
        max_pooled = sequence_embs.max(dim=1)[0]  # [batch_size, embed_dim]
        
        # 3. Attention pooling
        attention_scores = self.attention(sequence_embs)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # 应用mask到attention scores
        attention_scores = attention_scores.masked_fill(sequence_mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        attention_pooled = (sequence_embs * attention_weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, embed_dim]
        
        # 组合三种pooling结果
        combined_pooling = torch.cat([mean_pooled, max_pooled, attention_pooled], dim=-1)  # [batch_size, embed_dim*3]
        sequence_embed = self.sequence_pooling(combined_pooling)  # [batch_size, embed_dim]
        
        return sequence_embed

    def forward(self,
                user_id,
                item_id,
                gender,
                age_range,
                city,
                cluster_id,
                user_numeric,
                item_text_feat,
                user_sequence=None,
                sequence_mask=None):
        """
        Forward pass to compute matching score between user and item.
        """
        uid_embed = self.user_id_embedding(user_id)
        g_embed = self.gender_embedding(gender)
        a_embed = self.age_embedding(age_range)
        c_embed = self.city_embedding(city)
        cl_embed = self.cluster_embedding(cluster_id)
        num_proj = self.user_num_proj(user_numeric)

        # 处理用户历史序列特征
        if self.use_sequence and user_sequence is not None and sequence_mask is not None:
            seq_embed = self._process_user_sequence(user_sequence, sequence_mask)
            user_vector = torch.cat(
                [uid_embed, g_embed, a_embed, c_embed, cl_embed, num_proj, seq_embed], dim=-1
            )
        else:
            user_vector = torch.cat(
                [uid_embed, g_embed, a_embed, c_embed, cl_embed, num_proj], dim=-1
            )
        
        user_vector = self.user_tower(user_vector)

        item_embed = self.item_id_embedding(item_id)
        text_proj = self.item_text_proj(item_text_feat)

        item_vector = torch.cat([item_embed, text_proj], dim=-1)
        item_vector = self.item_tower(item_vector)

        # ✅ FAISS 检索相容：归一化向量后做余弦/点积
        user_vector = F.normalize(user_vector, p=2, dim=-1)
        item_vector = F.normalize(item_vector, p=2, dim=-1)

        scores = (user_vector * item_vector).sum(dim=-1)
        return scores

    def get_user_embedding(self, user_id, gender, age_range, city, cluster_id, user_numeric, 
                          user_sequence=None, sequence_mask=None):
        uid_embed = self.user_id_embedding(user_id)
        g_embed = self.gender_embedding(gender)
        a_embed = self.age_embedding(age_range)
        c_embed = self.city_embedding(city)
        cl_embed = self.cluster_embedding(cluster_id)
        num_proj = self.user_num_proj(user_numeric)

        # 处理用户历史序列特征
        if self.use_sequence and user_sequence is not None and sequence_mask is not None:
            seq_embed = self._process_user_sequence(user_sequence, sequence_mask)
            user_vector = torch.cat(
                [uid_embed, g_embed, a_embed, c_embed, cl_embed, num_proj, seq_embed], dim=-1
            )
        else:
            user_vector = torch.cat(
                [uid_embed, g_embed, a_embed, c_embed, cl_embed, num_proj], dim=-1
            )
        
        user_vector = self.user_tower(user_vector)
        return F.normalize(user_vector, p=2, dim=-1)  # ✅ 添加归一化

    def get_item_embedding(self, item_id, item_text_feat):
        item_embed = self.item_id_embedding(item_id)
        text_proj = self.item_text_proj(item_text_feat)
        item_vector = torch.cat([item_embed, text_proj], dim=-1)
        item_vector = self.item_tower(item_vector)
        return F.normalize(item_vector, p=2, dim=-1)  # ✅ 添加归一化
