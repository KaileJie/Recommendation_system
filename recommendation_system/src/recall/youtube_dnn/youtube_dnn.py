# src/recall/youtube_dnn/youtube_dnn.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class YoutubeDNN(nn.Module):
    """
    YouTube DNN: Multi-embedding user tower + numeric features + TF-IDF item features.
    """

    def __init__(self,
                 num_users,
                 num_items,
                 user_cate_dims,     # e.g. {"gender": 3, "age_range": 9, "city": 100, "cluster_id": 20}
                 user_numeric_dim,   # e.g. 3
                 item_text_dim,      # e.g. 300
                 embed_dim=64,
                 hidden_dims=(128, 64),
                 dropout=0.2):
        super(YoutubeDNN, self).__init__()

        # ---- Embedding layers ----
        self.user_id_embedding = nn.Embedding(num_users, embed_dim)
        self.gender_embedding = nn.Embedding(user_cate_dims["gender"], embed_dim)
        self.age_embedding = nn.Embedding(user_cate_dims["age_range"], embed_dim)
        self.city_embedding = nn.Embedding(user_cate_dims["city"], embed_dim)
        self.cluster_embedding = nn.Embedding(user_cate_dims["cluster_id"], embed_dim)

        # ---- User numeric projection ----
        self.user_num_proj = nn.Sequential(
            nn.Linear(user_numeric_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- Item embedding ----
        self.item_id_embedding = nn.Embedding(num_items, embed_dim)

        # ---- Item text projection ----
        self.item_text_proj = nn.Sequential(
            nn.Linear(item_text_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- User tower ----
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
                    f"user_cate_dims={user_cate_dims}")

    def forward(self,
                user_id,
                item_id,
                gender,
                age_range,
                city,
                cluster_id,
                user_numeric,
                item_text_feat):
        """
        Forward pass to compute matching score between user and item.
        """
        # ---- User side ----
        uid_embed = self.user_id_embedding(user_id)
        g_embed = self.gender_embedding(gender)
        a_embed = self.age_embedding(age_range)
        c_embed = self.city_embedding(city)
        cl_embed = self.cluster_embedding(cluster_id)
        num_proj = self.user_num_proj(user_numeric)

        user_vector = torch.cat(
            [uid_embed, g_embed, a_embed, c_embed, cl_embed, num_proj], dim=-1
        )
        user_vector = self.user_tower(user_vector)

        # ---- Item side ----
        item_embed = self.item_id_embedding(item_id)
        text_proj = self.item_text_proj(item_text_feat)

        item_vector = torch.cat([item_embed, text_proj], dim=-1)
        item_vector = self.item_tower(item_vector)

        # ---- Final dot product score ----
        scores = (user_vector * item_vector).sum(dim=-1)
        return scores

    def get_user_embedding(self, user_id, gender, age_range, city, cluster_id, user_numeric):
        uid_embed = self.user_id_embedding(user_id)
        g_embed = self.gender_embedding(gender)
        a_embed = self.age_embedding(age_range)
        c_embed = self.city_embedding(city)
        cl_embed = self.cluster_embedding(cluster_id)
        num_proj = self.user_num_proj(user_numeric)

        user_vector = torch.cat(
            [uid_embed, g_embed, a_embed, c_embed, cl_embed, num_proj], dim=-1
        )
        return self.user_tower(user_vector)

    def get_item_embedding(self, item_id, item_text_feat):
        item_embed = self.item_id_embedding(item_id)
        text_proj = self.item_text_proj(item_text_feat)
        item_vector = torch.cat([item_embed, text_proj], dim=-1)
        return self.item_tower(item_vector)
