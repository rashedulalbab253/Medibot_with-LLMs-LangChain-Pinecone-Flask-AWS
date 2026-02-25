"""
MediBot — Clinical Decision Support System
Industry-grade FastAPI application with RAG pipeline, conversation history,
streaming responses, health checks, and structured logging.
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
import os
import uvicorn
import logging
import time
import json
from datetime import datetime
from collections import deque
from typing import AsyncGenerator

# ─── Logging Setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("medibot")

# ─── App Initialization ──────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'medical-chatbot')
TOP_K = int(os.environ.get('RAG_TOP_K', '5'))
MAX_HISTORY = int(os.environ.get('MAX_CHAT_HISTORY', '10'))
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '1024'))

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Check your .env file.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

app = FastAPI(
    title="MediBot — Clinical Decision Support System",
    description="RAG-powered medical chatbot using Pinecone + Groq + LLaMA",
    version="2.0.0",
)

# CORS (allow all for local development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static/templates")

# ─── Model & Client Initialization (Lazy Loading) ──────────────────
embedding_model = None
pc = None
index = None
groq_client = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        logger.info("Initializing embedding model...")
        embedding_model = download_hugging_face_embeddings()
    return embedding_model

def get_pinecone_index():
    global pc, index
    if pc is None:
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
    return index

def get_groq_client():
    global groq_client
    if groq_client is None:
        logger.info("Initializing Groq client...")
        groq_client = Groq(api_key=GROQ_API_KEY)
    return groq_client

# Per-session chat history: session_id → deque of messages
chat_histories: dict[str, deque] = {}

# ─── Helper Functions ─────────────────────────────────────────────

def get_session_history(session_id: str) -> deque:
    """Get or create a conversation history for a session."""
    if session_id not in chat_histories:
        chat_histories[session_id] = deque(maxlen=MAX_HISTORY * 2)  # user+bot pairs
    return chat_histories[session_id]


def retrieve_context(query: str, top_k: int = TOP_K) -> tuple[str, list[dict]]:
    """
    Query Pinecone with the user's message and return formatted context + sources.
    """
    model = get_embedding_model()
    idx = get_pinecone_index()
    query_vector = model.encode(query).tolist()
    response = idx.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    context_parts = []
    sources = []
    seen_sources = set()

    matches = response.get("matches", [])
    for match in matches:
        meta = match.get("metadata", {})
        text = meta.get("text", "").strip()
        score = round(match.get("score", 0.0), 3)
        source = meta.get("source", "Unknown source")
        page = meta.get("page", "?")

        if text and score > 0.3:  # Only include relevant matches
            context_parts.append(f"[Relevance: {score}]\n{text}")
            source_key = f"{source}:p{page}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({"source": source, "page": page, "score": score})

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def build_messages(session_id: str, user_message: str, context: str) -> list[dict]:
    """
    Build the full message list including system prompt and conversation history.
    """
    history = get_session_history(session_id)
    formatted_prompt = system_prompt.replace("{context}", context if context else "No specific context retrieved. Provide general medical guidance.")

    messages = [{"role": "system", "content": formatted_prompt}]
    messages.extend(list(history))
    messages.append({"role": "user", "content": user_message})
    return messages


# ─── Routes ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index_route(request: Request):
    """Serve the main chat UI."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/get")
async def chat(
    request: Request,
    msg: str = Form(...),
    session_id: str = Form(default="default"),
):
    """
    Main chat endpoint.
    Retrieves context from Pinecone, builds prompt with history, calls Groq LLM.
    """
    if not msg or not msg.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    msg = msg.strip()
    start_time = time.time()
    logger.info(f"[Session: {session_id}] User: {msg[:80]}...")

    try:
        # 1. Retrieve relevant context from Pinecone
        context, sources = retrieve_context(msg)
        logger.info(f"Retrieved {len(sources)} relevant source(s).")

        # 2. Build messages with history
        messages = build_messages(session_id, msg, context)

        # 3. Call Groq LLM
        g_client = get_groq_client()
        completion = g_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,        # Lower = more factual/clinical
            top_p=0.9,
            frequency_penalty=0.1,  # Reduce repetition
        )

        answer = completion.choices[0].message.content
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"[Session: {session_id}] Response generated in {elapsed}s.")

        # 4. Update conversation history
        history = get_session_history(session_id)
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": answer})

        # 5. Return response with metadata
        return JSONResponse({
            "answer": answer,
            "sources": sources,
            "elapsed": elapsed,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"[Session: {session_id}] Error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "answer": "I apologize, I encountered an error processing your request. Please try again. If the issue persists, contact support.",
                "error": str(e),
                "sources": [],
            }
        )


@app.post("/clear-history")
async def clear_history(session_id: str = Form(default="default")):
    """Clear conversation history for a session."""
    if session_id in chat_histories:
        chat_histories[session_id].clear()
    logger.info(f"[Session: {session_id}] History cleared.")
    return JSONResponse({"status": "ok", "message": "Conversation history cleared."})


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and deployment."""
    try:
        idx = get_pinecone_index()
        stats = idx.describe_index_stats()
        vector_count = stats.get("total_vector_count", 0)
        return JSONResponse({
            "status": "healthy",
            "service": "MediBot Clinical Decision Support",
            "version": "2.0.0",
            "vector_count": vector_count,
            "model": GROQ_MODEL,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return JSONResponse({
        "name": "MediBot API",
        "version": "2.0.0",
        "endpoints": {
            "GET /": "Chat UI",
            "POST /get": "Send message (form: msg, session_id)",
            "POST /clear-history": "Clear chat history (form: session_id)",
            "GET /health": "Health check",
        }
    })


# ─── Entrypoint ──────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )