import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import pickle
from collections import namedtuple

from src.data.load_user_profile import load_user_profile_feature
from src.data.load_item_info import load_item_info
from src.data.load_behavior_log import iter_clean_behavior
from src.utils.weight_utils import apply_time_weight
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/youtube_dnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Sample = namedtuple("Sample", [
    "user_idx", "item_idx", "label", "gender", "age", "city", "cluster",
    "num_z", "item_text", "time_weight"])

class YoutubeDNNDataset(Dataset):
    def __init__(self,
                 user_profile_csv: str,
                 item_csv: str,
                 behavior_csv: str,
                 cutoff_date,
                 upper_date,
                 chunksize=50000,
                 num_negatives=4,
                 max_text_feat_dim=300,
                 time_decay_method: str = "linear",
                 decay_days: int = 30,
                 min_time_weight: float = 0.1,
                 filter_item_ids: set = None,
                 recall_mode: bool = False):

        self.users = load_user_profile_feature(user_profile_csv)
        self.items = load_item_info(item_csv)

        self.users["user_id"] = self.users["user_id"].astype(str)
        self.items["item_id"] = self.items["item_id"].astype(str)
        self.valid_item_ids = set(self.items["item_id"].unique())
        if filter_item_ids is not None:
            self.valid_item_ids &= filter_item_ids

        self.user2idx = {uid: i for i, uid in enumerate(self.users["user_id"].unique())}
        self.item2idx = {iid: i for i, iid in enumerate(self.items["item_id"].unique())}

        self.gender_vocab = sorted(self.users["gender"].unique())
        self.age_vocab = sorted(self.users["age_range"].unique())
        self.city_vocab = sorted(self.users["city"].astype(str).unique())
        self.cluster_vocab = sorted(self.users["cluster_id"].unique())

        self.gender2idx = {g: i for i, g in enumerate(self.gender_vocab)}
        self.age2idx = {a: i for i, a in enumerate(self.age_vocab)}
        self.city2idx = {c: i for i, c in enumerate(self.city_vocab)}
        self.cluster2idx = {c: i for i, c in enumerate(self.cluster_vocab)}

        self.num_cols = ["recency", "frequency", "actions_per_active_day_30d"]
        self.num_mean = self.users[self.num_cols].mean().values.astype(np.float32)
        self.num_std = self.users[self.num_cols].std(ddof=0).values.astype(np.float32)

        self.items["text"] = self.items["title"].fillna("") + " " + self.items["content"].fillna("")
        self.vectorizer = TfidfVectorizer(max_features=max_text_feat_dim)
        tfidf_matrix = self.vectorizer.fit_transform(self.items["text"].tolist())
        self.item_text_feats = {
            iid: tfidf_matrix[i].toarray()[0] for i, iid in enumerate(self.items["item_id"])
        }

        try:
            with open(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
                pickle.dump(self.vectorizer, f)
        except Exception as e:
            logger.warning(f"Failed to save vectorizer: {e}")

        if not self.item_text_feats:
            logger.warning("TF-IDF features are empty! Using zero vectors instead.")
            self.item_text_feats = {
                iid: np.zeros(max_text_feat_dim, dtype=np.float32) for iid in self.item2idx
            }

        # ✅ recall_mode 自动填充时间范围
        if recall_mode:
            cutoff_date = pd.to_datetime("2000-01-01")
            upper_date = pd.to_datetime("2100-01-01")

        interactions = []
        for chunk in iter_clean_behavior(
            behavior_csv,
            chunksize=chunksize,
            cutoff_date=cutoff_date,
            upper=upper_date,
            extra_usecols=["item_id"]):

            chunk["user_id"] = chunk["user_id"].astype(str)
            chunk["item_id"] = chunk["item_id"].astype(str)
            chunk = chunk[chunk["item_id"].isin(self.valid_item_ids)]
            chunk = apply_time_weight(
                chunk,
                dt_col="dt",
                method=time_decay_method,
                ref_date=cutoff_date,
                decay_days=decay_days,
                min_weight=min_time_weight)

            interactions.append(chunk[["user_id", "item_id", "time_weight"]])

        if not interactions:
            logger.error("No interactions found.")
            self.samples = []
            self._user_feat_cache = {}  # ✅ fallback empty cache
            return

        self.interactions = pd.concat(interactions, ignore_index=True)
        valid_users = set(self.user2idx.keys())
        valid_items = set(self.item2idx.keys())
        self.interactions = self.interactions[
            self.interactions["user_id"].isin(valid_users) &
            self.interactions["item_id"].isin(valid_items)
        ].reset_index(drop=True)

        self.num_negatives = num_negatives
        self.all_items = list(self.item2idx.keys())
        self.user2items = self.interactions.groupby("user_id")["item_id"].apply(set).to_dict()

        self._user_feat_cache = {}
        for row in self.users.itertuples(index=False):
            uid = str(row.user_id)
            if uid not in self.user2idx:
                continue
            gender_idx = self.gender2idx.get(row.gender, 0)
            age_idx = self.age2idx.get(row.age_range, 0)
            city_idx = self.city2idx.get(row.city, 0)
            cluster_idx = self.cluster2idx.get(row.cluster_id, 0)

            num_vec = np.array([
                row.recency, row.frequency, row.actions_per_active_day_30d
            ], dtype=np.float32)
            num_z = (num_vec - self.num_mean) / (self.num_std + 1e-8)

            self._user_feat_cache[uid] = (
                gender_idx, age_idx, city_idx, cluster_idx, num_z
            )

        self.samples = self._generate_samples()

    def _sample_negatives(self, user_id, num_samples):
        seen = self.user2items.get(user_id, set())
        candidates = list(set(self.all_items) - seen)
        if not candidates:
            return []
        weights = torch.ones(len(candidates))
        indices = torch.multinomial(weights, num_samples, replacement=len(candidates) < num_samples)
        return [candidates[i] for i in indices.tolist()]

    def _generate_samples(self):
        samples = []
        for row in self.interactions.itertuples(index=False):
            samples.append((row.user_id, row.item_id, 1, row.time_weight))
            
            if self.num_negatives > 0:
                for neg_item in self._sample_negatives(row.user_id, self.num_negatives):
                    samples.append((row.user_id, neg_item, 0, 1.0))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, item_id, label, time_weight = self.samples[idx]
        uidx = self.user2idx[user_id]
        iidx = self.item2idx[item_id]

        gender_idx, age_idx, city_idx, cluster_idx, num_z = self._user_feat_cache.get(
            user_id, (0, 0, 0, 0, np.zeros(len(self.num_cols), dtype=np.float32))
        )
        item_text = self.item_text_feats.get(
            item_id, np.zeros_like(next(iter(self.item_text_feats.values())))
        )

        return Sample(
            user_idx=torch.tensor(uidx, dtype=torch.long),
            item_idx=torch.tensor(iidx, dtype=torch.long),
            label=torch.tensor(label, dtype=torch.float),
            gender=torch.tensor(gender_idx, dtype=torch.long),
            age=torch.tensor(age_idx, dtype=torch.long),
            city=torch.tensor(city_idx, dtype=torch.long),
            cluster=torch.tensor(cluster_idx, dtype=torch.long),
            num_z=torch.tensor(num_z, dtype=torch.float),
            item_text=torch.tensor(item_text, dtype=torch.float),
            time_weight=torch.tensor(time_weight, dtype=torch.float)
        )

    def get_user_categorical_vocab_sizes(self):
        return {
            "gender": len(self.gender_vocab),
            "age_range": len(self.age_vocab),
            "city": len(self.city_vocab),
            "cluster_id": len(self.cluster_vocab),
        }

    def get_user_numeric_norm(self):
        return {
            "mean": self.num_mean,
            "std": self.num_std,
            "cols": self.num_cols
        }

    def get_item_text_dim(self):
        return len(next(iter(self.item_text_feats.values())))

    def get_num_users_items(self):
        return len(self.user2idx), len(self.item2idx)

    def get_user_features(self, user_id):
        return self._user_feat_cache.get(
            user_id, (0, 0, 0, 0, np.zeros(len(self.num_cols), dtype=np.float32))
        )
