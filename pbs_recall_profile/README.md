# PBS News Keyword Recall System (v5.0)

A scalable keyword extraction and recall system built on top of the PBS NewsHour economy articles. This project enables automatic keyword extraction (TF-IDF, TextRank, Fusion), inverted index recall, and a FastAPI-based recall API for frontend/backend integration.

---

## Features

- ✅ **Automated Keyword Extraction**
  - TF-IDF-based keywords
  - TextRank-based keywords
  - Fusion strategy combining both
- ✅ **Inverted Index-based Recall**
  - Supports `tfidf`, `textrank`, `final`, or `simple` keyword strategy
- ✅ **FastAPI Web Server**
  - `/recall` endpoint for keyword search with recall strategy

---

## Project Structure

```
pbs_spider/
├── api/                       # FastAPI REST API
│   └── recall_api.py
│
├── dao/                       # MongoDB DAO layer
│   └── mongo_db.py
│
├── recall/                    # Keyword extraction and recall logic
│   ├── data_loader.py
│   ├── keyword_updater.py
│   ├── recall_service.py
│   └── tfidf_model.py
│
├── spiders/                   # Scrapy spider for PBS articles
│   └── pbs_economy.py
│
├── utils/                     # Utility functions
│   └── text.py
│
├── settings.py                # Scrapy project settings
├── run_pbs.sh                 # Bash script for scheduling
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── scrapy.cfg                 # Scrapy config
```

---

## How It Works

1. **Keyword Extraction (`keyword_updater.py`)**
   - Loads recent articles from MongoDB
   - Extracts keywords using:
     - TF-IDF
     - TextRank
     - Combined strategy (Fusion)
   - Saves keywords to:
     - `tfidf_keywords`
     - `textrank_keywords`
     - `final_keywords`

2. **Recall Service (`recall_service.py`)**
   - Builds inverted index for each strategy
   - Supports recall via `simple`, `final`, `tfidf`, `textrank`

3. **API (`recall_api.py`)**
   - `/recall` endpoint supports querying keywords using strategy and top-n

---

## 🛠 Setup

### 1. Environment Setup

```bash
conda create -n pbs python=3.9
conda activate pbs
pip install -r requirements.txt
```

### 2. MongoDB

Ensure MongoDB is running locally or update URI in `mongo_db.py`.

---

## Run Keyword Extraction

```bash
python -m pbs_spider.recall.keyword_updater
```

---

## Run FastAPI Server

```bash
uvicorn pbs_spider.api.recall_api:app --reload
```

Example request:
```
GET http://127.0.0.1:8000/recall?keywords=health&strategy=tfidf&topn=5
```

---

## License

MIT License.
