# masters_thesis_project

---

# 🌤️ Weather-Aware Event Recommendation System

A context-rich, two-stage recommendation system that personalizes event suggestions based on user preferences, geographic proximity, and real-time weather conditions. This project combines deep retrieval (two-tower model) with gradient-boosted ranking (CatBoost) and integrates with Hopsworks and Streamlit for deployment and visualization.

---

## 📌 Table of Contents

* [✨ Features](#-features)
* [📊 Datasets](#-datasets)
* [⚙️ System Architecture](#-system-architecture)
* [🏗️ Implementation Overview](#-implementation-overview)
* [🧪 Evaluation](#-evaluation)
* [🚀 Deployment](#-deployment)
* [📺 Streamlit Demo](#-streamlit-demo)
* [🛠 Setup Instructions](#-setup-instructions)
* [📁 Project Structure](#-project-structure)
* [🧠 Future Work](#-future-work)
* [📄 License](#-license)

---

## ✨ Features

* 🔍 **Two-Tower Neural Retrieval**: Scalable deep retrieval using user and event embeddings trained with TensorFlow Recommenders.
* 📊 **CatBoost Ranking Model**: Learns personalized ranking using event metadata, user context, interaction distance, and weather data.
* 🌤️ **Weather-Aware Recommendations**: Incorporates real-time weather at both event and user locations.
* 🧊 **Cold-Start Ready**: Robust retrieval for new users or events using content-based embeddings.
* 🔌 **Hopsworks Feature Store Integration**: Efficient feature retrieval and model serving.
* 🖥️ **Streamlit Frontend**: Interactive app for exploring personalized recommendations in real-time.

---

## 📊 Datasets

This project supports both synthetic and semi-real datasets:

### Synthetic Dataset

* 50,000 users, 10,000 events, 500,000 interactions
* Geographically distributed users and events
* Weather and location-based features simulated using `geopy`, `Faker`, and probabilistic models

### Semi-Real Dataset

* Based on a real Kaggle event:  https://www.kaggle.com/competitions/event-recommendation-engine-challenge/data?select=event_attendees.csv.gz
* 4,967 events, 5,315 users, 28,249 interactions
* Enriched with weather and location metadata using external APIs

---

## ⚙️ System Architecture

```
              ┌─────────────┐
              │ User Query  │
              └─────┬───────┘
                    │
            ┌───────▼────────┐
            │  Two-Tower     │
            │  Retrieval     │ <───── Embeddings (TF Recommenders)
            └───────┬────────┘
                    │
     Top-K Event IDs│
            ┌───────▼────────────┐
            │ Feature Store      │
            └───────┬────────────┘
                    │
            ┌───────▼────────┐
            │ CatBoost Ranker│ <───── Weather + Geo + User features
            └───────┬────────┘
                    │
            ┌───────▼───────┐
            │ Recommendations │
            └───────────────┘
```

---

## 🏗️ Implementation Overview

* `data_generation/`: Generate synthetic and semi-real datasets
* `retrieval/`: Two-tower model implementation using TensorFlow Recommenders
* `ranking/`: CatBoost training pipeline with weather-aware and non-weather variants
* `hopsworks_deployment/`: Deployment logic for feature views, models, and endpoints
* `streamlit_app/`: UI for live recommendations with weather toggles
* `notebooks/`: Training, evaluation, and exploration notebooks

---

## 🧪 Evaluation

| Metric                | Weather-Aware | No-Weather | Hybrid (Baseline) |
| --------------------- | ------------- | ---------- | ----------------- |
| Precision\@1          | **0.752**     | 0.725      | 0.687             |
| NDCG                  | **0.776**     | 0.742      | 0.688             |
| Recall\@10            | 0.476         | **0.512**  | 0.418             |
| Cold-Start Robustness | ✅             | ✅          | ❌                 |

> 📌 Weather-aware model improves top-k precision and ranking quality at small k, but may slightly reduce recall due to context filtering.

---

## 🚀 Deployment

* Models and features are deployed via [Hopsworks](https://www.hopsworks.ai/).
* `event_ranking_transformer.py` and `event_ranking_predictor.py` are used to serve the ranking model.
* Query model and embedding index deployed using TensorFlow Serving and FAISS.

---

## 📺 Streamlit Demo

Launch the Streamlit app:

```bash
cd streamlit_app
streamlit run app.py
```

### Features:

* Select or search for a user profile
* Toggle weather-aware filtering
* Adjust distance threshold slider
* View ranked events with scores

---

## 🛠 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/AbigailUchennaNkama/masters_thesis_project.git
cd weather-event-recommender
```

### 2. Create environment

```bash
conda create -n eventrec python=3.10
conda activate eventrec
pip install -r requirements.txt
```

### 3. Authenticate to Hopsworks

```python
import hopsworks
project = hopsworks.login()
```

### 4. Train & deploy


---



