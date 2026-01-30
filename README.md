# 🧠 Medibot — RAG-Based Clinical Decision Support System

Medibot is a **Retrieval-Augmented Generation (RAG)** powered clinical decision support chatbot that provides **domain-grounded, reliable medical answers** by constraining Large Language Models (LLMs) to curated medical literature.

This project emphasizes **hallucination reduction, safety, system reliability, and production-grade deployment**, making it suitable for **AI/ML Engineer roles** and **PhD-level research portfolios**.

---

## 🚀 Key Features
- Domain-constrained medical question answering using RAG  
- Reduced hallucinations via document-grounded generation  
- Domain-constrained medical question answering using RAG  
- Reduced hallucinations via document-grounded generation  
- High-performance asynchronous backend with **FastAPI**
- Scalable vector search using **Pinecone**
- Automated CI/CD for Docker image builds
- Healthcare safety–aware system design  

---

## 🎯 Motivation
Large Language Models demonstrate strong reasoning abilities but are **unreliable in high-risk domains such as healthcare**, where hallucinated responses can cause serious harm.

This project investigates whether **retrieval grounding and architectural constraints** can:
- Improve factual correctness  
- Reduce hallucinations  
- Enable safe clinical decision support **without model fine-tuning**

---

## 🧠 Research Objective
To study the effectiveness of **Retrieval-Augmented Generation** in improving reliability and safety of LLM-based medical question answering systems when restricted to **domain-specific clinical documents**.

---

## 🏗️ System Architecture

### High-Level RAG Pipeline
1. Medical PDFs are ingested and chunked  
2. Text chunks are embedded using Sentence Transformers  
3. Embeddings are stored in Pinecone Vector Database  
4. User queries are embedded and matched via similarity search  
5. Retrieved context is injected into the LLM prompt  
6. **Groq (Llama 3.3 70B)** generates a **context-grounded clinical response**


---

## 🧪 Methodology

### Data Ingestion & Preprocessing
- PDF documents loaded using `DirectoryLoader` and `PyPDFLoader`
- Recursive character splitting  
- Chunk size: 500 characters  
- Overlap: 20 characters  
- Optimized for semantic continuity

### Vector Representation
- Embedding model: `all-MiniLM-L6-v2`
- 384-dimensional dense embeddings
- Cosine similarity–based retrieval

### Retrieval-Augmented Generation
- Top-k relevant document chunks retrieved from Pinecone
- Context injected directly into the LLM prompt
- Prevents speculative or open-ended generation

---

## 📊 Evaluation Summary
- Low-latency medical information retrieval  
- Improved factual grounding compared to vanilla prompting  
- Responses constrained strictly to provided medical literature  

> This project prioritizes **reliability, interpretability, and safety** over generative creativity.

---

## 🛠️ Tech Stack

### Core Technologies
- **Language:** Python 3.12
- **LLM:** Groq (Llama 3.3 70B)
- **Framework:** LangChain  
- **Vector Database:** Pinecone  
- **Embeddings:** Sentence Transformers  
- **Backend:** FastAPI  

### Deployment & MLOps
- Docker  
- Docker Hub
- GitHub Actions (CI/CD)  

---

## 🔮 Future Work

- **Multimodal Clinical RAG**  
  Extend the system to support medical images (X-ray, MRI, CT) by integrating vision encoders and multimodal retrieval for image–text grounded clinical reasoning.

- **Physician-in-the-Loop Validation**  
  Introduce expert feedback loops where clinicians can validate, correct, or annotate responses to improve system reliability and trustworthiness.

- **Automated Knowledge Base Expansion**  
  Enable continuous synchronization with external medical sources such as PubMed, clinical trial registries, and treatment guidelines using scheduled ingestion pipelines.

- **Advanced Retrieval Strategies**  
  Explore hybrid retrieval techniques combining dense vectors with sparse methods (BM25) and reranking models to further improve context relevance.

- **Comparative RAG vs Fine-Tuning Study**  
  Conduct empirical studies comparing RAG-based grounding against domain-specific LLM fine-tuning in terms of accuracy, hallucination rate, and computational cost.

- **Clinical Safety & Compliance Layer**  
  Incorporate rule-based safety checks and policy validation aligned with healthcare standards (e.g., HIPAA/GDPR) to support real-world clinical deployment.


# ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/rashedulalbab253/Medibot_with-LLMs-LangChain-Pinecone-Flask-AWS.git
cd Medibot_with-LLMs-LangChain-Pinecone-Flask-AWS
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
PINECONE_API_KEY = "your_pinecone_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

### 5. Ingest Data
```bash
# Store embeddings to Pinecone
python store_index.py
```

### 6. Run Application
```bash
python app.py
```
Open `http://localhost:8080` in your browser.

---

# 🐳 Docker Support

### Build Image
```bash
docker build -t medibot .
```

### Run Container
```bash
docker run -d -p 8080:8080 --name medibot -e PINECONE_API_KEY="your_api_key" -e GROQ_API_KEY="your_api_key" medibot
```

---

# 🚀 CI/CD Pipeline
This repository uses **GitHub Actions** to automatically build and push the Docker image to **Docker Hub** on every push to the `main` branch.

### Required GitHub Secrets:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub Access Token
- `PINECONE_API_KEY`: Your Pinecone API Key
- `GROQ_API_KEY`: Your Groq API Key

**Image Registry:** `rashedulalbab1234/medibot:latest`

---
**Author:** Rashedul Albab
