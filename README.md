# 🧠 Personalized CV QnA Bot

A production-ready, AI-powered Question Answering (QnA) system that uses your own CV as the knowledge base.

Built with:
- 🧠 FAISS for semantic search
- ⚡ FastAPI for serving answers via API
- 🎯 MLflow for experiment tracking
- ⚙️ Apache Airflow for automated index building
- 🐳 Docker & Docker Compose for full-stack orchestration

---

## 🔀 Branch Overview

| Branch                  | Description                                            |
|-------------------------|--------------------------------------------------------|
| `main`                  | 🧪 Minimal FastAPI + FAISS setup (no MLflow / Airflow) |
| `mlflow_integration`    | 🎯 Adds MLflow experiment tracking                     |
| `airflow_integration`   | 💼 Full-stack: Airflow + MLflow + QnA app in sync      |

---

## 🚀 Quickstart (Airflow + MLflow + QnA)

> Only for `airflow_integration` branch.

### 🧩 Prerequisites

- Docker & Docker Compose installed
- Python 3.10 (for local testing only)

### 🏗️ Clone and Launch

```bash
git clone https://github.com/your-username/Personalised_CV-QnA-BOT.git
cd Personalised_CV-QnA-BOT
git checkout airflow_integration
docker-compose up -d --build
```

> ✅ Airflow will auto-initialize the database and create an admin user (`admin`/`admin`)

---

## 📁 Folder Structure

```
.
├── app/                    # FastAPI application
│   ├── main.py             # API entry point
│   └── qa/                 # Core ML logic (embedding, FAISS, retriever)
├── scripts/
│   └── build_index.py      # Script to build FAISS index from CV
├── airflow/
│   └── dags/               # Airflow DAGs
├── data/                   # Shared folder for index + CV document
│   └── docs/               # Input documents (e.g. your CV)
├── mlruns/                 # MLflow experiment storage
├── Dockerfile              # QnA app Dockerfile
├── Dockerfile.airflow      # Airflow image with custom dependencies
├── requirements.txt        # Python dependencies
└── docker-compose.yml      # Multi-container setup
```

---

## 🔌 APIs

### ❓ Ask a Question

```http
POST /ask?question=What did Narendar do in NLP?
```

### 🔁 Reload FAISS Index

```http
POST /reload-index
```

Reloads `index.faiss` and `doc_map.pkl` from `/app/data`.

---

## ⚙️ Airflow Pipeline

> [http://localhost:8080](http://localhost:8080)  
Login: `admin` / `admin`

The DAG:
- Rebuilds FAISS index daily
- Saves `index.faiss` and `doc_map.pkl` into `./data`
- (Optional) Triggers `/reload-index` in QnA app

---

## 📊 MLflow Tracking

> [http://localhost:5050](http://localhost:5050)

Logs:
- Index build times
- Document chunk counts
- Artifacts like FAISS index and doc_map

---

## 💡 To Do Next

- [ ] Add Streamlit UI
- [ ] Enable multiple document support
- [ ] Integrate email or Slack notifications from Airflow
- [ ] Deploy to AWS / GCP

---

## 👨‍💻 Author

**Narendar Punithan**  
_Data Scientist | ML Engineer | Generative AI Enthusiast_  
📍 London, United Kingdom  
📫 akashnarendar2013@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/narendar-punithan)

---
