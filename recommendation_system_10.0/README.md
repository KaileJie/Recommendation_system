# 📌 Recommendation System v10.0

## 🚀 Overview
This project implements a **two-stage recommendation system** (**Recall → Rank**) with significant improvements in **version 10.0**.

Pipeline:
1. **Recall Layer**
   - Multi-strategy recall: `ItemCF`, `Keyword-Hybrid`, `YouTube DNN (Faiss)`
   - Generate candidate items for each user

2. **Ranking Layer**
   - Introduced **Meta-Ranker (GBDT + LR)**
   - Fuse recall scores with user profile features
   - Output final Top-K recommendations

Evaluation Metrics: **Recall@20, Precision@20, AUC**

---

## 📂 Project Structure
```bash
recommendation_system/
├── src/
│   ├── recall/
│   │   ├── item_cf/                  # ItemCF recall
│   │   ├── keyword/                  # Keyword (TF-IDF, TextRank, Hybrid)
│   │   ├── youtube_dnn/              # YouTube DNN + Faiss
│   │   │   ├── train_youtube_dnn.py
│   │   │   ├── faiss_index_youtube_dnn.py
│   │   │   ├── recall_youtube_dnn_faiss.py
│   │   │   └── evaluate_youtube_dnn.py
│   ├── rank/
│   │   └── meta_ranker/              # Meta-Ranker
│   │       ├── build_training_data.py
│   │       ├── train_meta_ranker.py
│   │       └── infer_meta_ranker.py
│   ├── data/                         # Data loading utilities
│   └── utils/                        # Shared tools
│
├── data/
│   ├── raw/                          # Raw logs & item metadata
│   └── processed/                    # User profiles
│
├── output/
│   ├── item_cf/                      # ItemCF recall results
│   ├── keyword/                      # Keyword recall results
│   ├── youtube_dnn/                  # YouTube DNN models & recall
│   └── meta_ranker/                  # Meta-Ranker training & inference
│
└── README.md
```

---

## 🆕 What’s New in v10.0

### 1. Unified Data Splitting
- **Training window**: `TRAIN_WINDOW_DAYS = 30`
- **Evaluation window**: `CUTOFF_DAYS = 7`
- All modules follow **30 days training + last 7 days testing**.

### 2. Recall Layer
- **ItemCF**
  - Updated dataset logic with new window.
  - Metrics upgraded to Recall@20, Precision@20.
- **Keyword**
  - Implemented three strategies: TF-IDF, TextRank, Hybrid.
  - **Hybrid** is used as the main recall output.
- **YouTube DNN**
  - Support valid item filtering.
  - Introduced time-decay weighting.
  - FAISS-based fast recall.
  - Unified evaluation: Recall@50, Precision@50.

### 3. Ranking Layer: Meta-Ranker (GBDT + LR)
- **Training data construction**
  - Merge recall results from ItemCF, Keyword-Hybrid, YouTube DNN.
  - Ground truth: last 7 days of user clicks.
  - Negative sampling (5:1 ratio).
  - Add user profile features:
    - `age_range, gender, city, cluster_id`
    - `recency, frequency, actions_per_active_day_30d`
    - `monetary, rfm_score`
- **Training**
  - GBDT (LightGBM) learns feature interactions.
  - Logistic Regression trained on GBDT-leaf encodings.
  - Models saved: `gbdt_model.pkl`, `lr_model.pkl`, `onehot_encoder.pkl`.
- **Inference**
  - Use GBDT+LR to re-rank recall candidates.
  - Output **Top-K JSON recommendations**.

---

## ⚙️ How to Run

### Step 1: Prepare Data
```
data/raw/user_behavior_log_info.csv
data/raw/item_metadata.csv
data/processed/user_profile_feature.csv
```

### Step 2: Run Recall
```bash
# ItemCF
python src/recall/item_cf/itemcf_recall.py

# Keyword Hybrid
python src/recall/keyword/keyword_hybrid_recall.py

# YouTube DNN
python src/recall/youtube_dnn/train_youtube_dnn.py
python src/recall/youtube_dnn/faiss_index_youtube_dnn.py
python src/recall/youtube_dnn/recall_youtube_dnn_faiss.py
```

### Step 3: Build Training Data
```bash
python src/rank/meta_ranker/build_training_data.py
```

### Step 4: Train Meta-Ranker
```bash
python src/rank/meta_ranker/train_meta_ranker.py
```

### Step 5: Run Inference
```bash
python src/rank/meta_ranker/infer_meta_ranker.py
```

---

## 📊 Output Files

- **Recall Layer**
  - `output/item_cf/itemcf_recall.csv`
  - `output/keyword/keyword_hybrid_recall.csv`
  - `output/youtube_dnn/youtube_dnn_faiss_recall.csv`

- **Meta-Ranker**
  - `output/meta_ranker/training_data.json`
  - `output/meta_ranker/gbdt_model.pkl`
  - `output/meta_ranker/lr_model.pkl`
  - `output/meta_ranker/onehot_encoder.pkl`
  - `output/meta_ranker/meta_ranker_infer.json`


---

## 📈 Evaluation
- **Recall@20**
- **Precision@20**
- **AUC (Meta-Ranker validation)**

---

✅ This version (v10.0) completes the **Recall + Rank two-stage framework**, with GBDT+LR fusion ensuring stronger personalization and better ranking performance.
