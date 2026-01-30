# 🧠 Medibot — RAG-Based Clinical Decision Support System

Medibot is a **Retrieval-Augmented Generation (RAG)** powered clinical decision support chatbot that provides **domain-grounded, reliable medical answers** by constraining Large Language Models (LLMs) to curated medical literature.

This project emphasizes **hallucination reduction, safety, system reliability, and production-grade deployment**, making it suitable for **AI/ML Engineer roles** and **PhD-level research portfolios**.

---

## 🚀 Key Features
- Domain-constrained medical question answering using RAG  
- Reduced hallucinations via document-grounded generation  
- Modular and scalable architecture  
- Dockerized and deployed on AWS  
- CI/CD using GitHub Actions  
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
6. LLM generates a **context-grounded clinical response**


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

- **Uncertainty Estimation & Confidence Scoring**  
  Add confidence estimation mechanisms to quantify uncertainty in responses and trigger Human-in-the-Loop intervention when confidence is low.

- **Scalability & Performance Optimization**  
  Optimize indexing, caching, and parallel retrieval to support large-scale document collections and high-concurrency clinical usage.

- **Explainability & Evidence Tracing**  
  Enhance transparency by explicitly linking generated answers to supporting document passages for improved interpretability and auditability.






# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/rashedulalbab253/Medibot_with-LLMs-LangChain-Pinecone-Flask-AWS.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.12 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & Groq credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:8080
```


### Techstack Used:

- Python
- LangChain
- FastAPI
- Groq (Llama 3.3 70B)
- Pinecone



# Docker CI/CD with Github Actions

## 1. Setup Docker Hub
    - Create a Docker Hub account.
    - Create a repository named `medibot`.

## 2. Setup github secrets:

   - `DOCKER_USERNAME`: Your Docker Hub username (`rashedulalbab1234`)
   - `DOCKER_PASSWORD`: Your Docker Hub Personal Access Token
   - `PINECONE_API_KEY`: Your Pinecone API Key
   - `GROQ_API_KEY`: Your Groq API Key

## 3. Deployment
    - Every push to the `main` branch will automatically build and push a new Docker image to `rashedulalbab1234/medibot:latest`.

Author:Rashedul Albab
