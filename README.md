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
- **Language:** Python 3.10  
- **LLM:** OpenAI GPT  
- **Framework:** LangChain  
- **Vector Database:** Pinecone  
- **Embeddings:** Sentence Transformers  
- **Backend:** Flask  

### Deployment & MLOps
- Docker  
- AWS EC2  
- AWS ECR  
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
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - OPENAI_API_KEY

Author:Rashedul Albab
