# 🧠 MediBot v2.0 — Production-Grade Clinical Decision Support System

MediBot is an advanced **Retrieval-Augmented Generation (RAG)** powered clinical decision support chatbot. It is designed to provide **domain-grounded, reliable medical answers** by constraining Large Language Models (LLMs) to curated medical literature.

This project emphasizes **hallucination reduction, safety, system reliability, and state-of-art UI/UX**, making it suitable for both **AI/ML Engineer portfolios** and **clinical research environments**.

---

## 🚀 What's New in v2.0
- **Architectural Overhaul:** Transitioned from Flask to a high-performance **FastAPI** asynchronous backend.
- **Premium User Interface:** A brand new, fully responsive, clinical-grade **Light Glassmorphism Theme** with interactive components, FontAwesome icons, and typing animations.
- **Conversational Memory:** Implemented session-based chat history, allowing the AI to understand multi-turn medical inquiries.
- **Advanced RAG Pipeline:** Intelligent context retrieval filtering (relevance score > 0.3) and precise source citations (document & page mapping) directly in the UI.
- **Optimized Data Ingestion:** Sentence-aware chunking with MD5 hash deduplication for cleaner vector stores, processed natively through **Pinecone**.
- **Production-Ready Docker:** Upgraded Dockerfile with optimized layer caching, system dependencies, and integrated application health checks.

---

## 🎯 Motivation
Large Language Models demonstrate strong reasoning abilities but are **unreliable in high-risk domains such as healthcare**, where hallucinated responses can cause serious harm.

This project investigates how **retrieval grounding, architectural constraints, and robust system prompts** can:
- Improve factual correctness  
- Eliminate speculative generation  
- Enable safe clinical decision support **without model fine-tuning**

---

## 🏗️ System Architecture

### High-Level RAG Pipeline
1. **Ingestion:** Medical PDFs are recursively parsed, cleaned, and split using sentence-boundary-aware chunking.
2. **Embedding:** Text chunks are batch-encoded into 384-dimensional dense vectors using `SentenceTransformers`.
3. **Storage:** Vectors and serializable metadata are upserted into a **Pinecone Serverless Vector Database**.
4. **Retrieval:** User queries are embedded, matched via cosine similarity, and strictly filtered.
5. **Generation:** **Groq (LLaMA 3.3 70B)** receives the user query, chat history, and context to generate a safely constrained clinical response.

---

## 🧪 Methodology

### Data Preprocessing
- **Source:** Directory-based PDF extraction.
- **Chunking Strategy:** Recursive character splitting.
  - Chunk size: `800` characters  
  - Overlap: `150` characters  
- **Quality Control:** Built-in text cleaning and node deduplication via MD5 hashing.

### Vector Representation
- **Model:** `all-MiniLM-L6-v2`
- **Metric:** Cosine similarity.

### RAG Constraints & Safety
- **Prompt Engineering:** Clinical-grade persona prompt with strict instructions to route emergencies and decline non-medical questions.
- **Context Filtering:** Only fragments with a similarity score `> 0.3` are injected into the prompt.
- **Source Transparency:** All bot responses natively display the exact source document to the user.

---

## 🛠️ Tech Stack

### Core Technologies
- **Language:** Python 3.12+
- **LLM Engine:** Groq API (LLaMA 3.3 70B Versatile)
- **Vector Database:** Pinecone (Serverless)
- **Embeddings:** HuggingFace `sentence-transformers`
- **Backend Framework:** FastAPI & Uvicorn
- **Frontend:** Vanilla JS, HTML5, Modern CSS (CSS Variables, Flexbox)

### Deployment & MLOps
- **Containerization:** Docker
- **Repository:** Docker Hub
- **CI/CD:** GitHub Actions (Automated building & pushing)

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/rashedulalbab253/Clinical-Decision-Support-System-RAG-Powered-Medical-Chatbot.git
cd Clinical-Decision-Support-System-RAG-Powered-Medical-Chatbot
```

### 2. Create environment
```bash
conda create -n medibot python=3.12 -y
conda activate medibot
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory:
```ini
PINECONE_API_KEY="your_pinecone_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
```

### 5. Ingest Data
Process your local PDFs and build the Pinecone index:
```bash
python store_index.py
```

### 6. Run Application
Start the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```
Open `http://localhost:8080` in your web browser.

---

## 🐳 Docker Support

### Build Image Locally
```bash
docker build -t medibot .
```

### Run Container
```bash
docker run -d -p 8080:8080 --name medibot -e PINECONE_API_KEY="your_api_key" -e GROQ_API_KEY="your_api_key" medibot
```

---

## 🚀 CI/CD Pipeline
This repository uses **GitHub Actions** to automatically build and push the Docker image to **Docker Hub** on every push to the `main` branch.

### Required GitHub Repository Secrets:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub Access Token
- *(Optional)* Pinecone and Groq keys can be passed as runtime variables.

**Image Registry:** `rashedulalbab1234/medibot:latest`

---
**Designed & Developed by Rashedul Albab**
