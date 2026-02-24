from dotenv import load_dotenv
import os
import logging
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings, batch_encode
from pinecone import Pinecone, ServerlessSpec
import time

# ─── Logging Setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Check your .env file.")

# ─── Load & Process Documents ────────────────────────────────────
logger.info("=" * 60)
logger.info("MediBot Index Builder — Starting")
logger.info("=" * 60)

logger.info("Step 1/5: Loading PDF data...")
extracted_data = load_pdf_file(data_dir='data/')
if not extracted_data:
    raise RuntimeError("No PDF documents found in 'data/' directory. Please add PDF files and retry.")
logger.info(f"Loaded {len(extracted_data)} pages.")

logger.info("Step 2/5: Filtering metadata...")
filter_data = filter_to_minimal_docs(extracted_data)

logger.info("Step 3/5: Splitting into chunks...")
text_chunks = text_split(filter_data, chunk_size=800, chunk_overlap=150)
logger.info(f"Total unique chunks created: {len(text_chunks)}")

# ─── Load Embedding Model ────────────────────────────────────────
logger.info("Step 4/5: Loading embedding model...")
model = download_hugging_face_embeddings()

# ─── Connect to Pinecone ─────────────────────────────────────────
logger.info("Step 5/5: Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Create index if it doesn't exist
existing_indexes = [idx.name for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    logger.info(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    logger.info("Waiting for index to be ready...")
    while True:
        index_info = pc.describe_index(index_name)
        if index_info.status["ready"]:
            break
        time.sleep(2)
    logger.info("Index is ready.")
else:
    logger.info(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)

# ─── Generate Embeddings in Batches ──────────────────────────────
logger.info(f"Generating embeddings for {len(text_chunks)} chunks...")
texts = [chunk["page_content"] for chunk in text_chunks]
all_embeddings = batch_encode(model, texts, batch_size=64)
logger.info(f"Generated {len(all_embeddings)} embeddings.")

# ─── Prepare Vectors ─────────────────────────────────────────────
vectors = []
for i, (chunk, embedding) in enumerate(zip(text_chunks, all_embeddings)):
    metadata = {k: v for k, v in chunk["metadata"].items() if v is not None and isinstance(v, (str, int, float, bool))}
    metadata["text"] = chunk["page_content"][:1000]  # Pinecone metadata value limit safety

    vectors.append({
        "id": f"med_{i:06d}",
        "values": embedding,
        "metadata": metadata,
    })

# ─── Upsert Vectors in Batches ───────────────────────────────────
BATCH_SIZE = 100
total_batches = (len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE
logger.info(f"Upserting {len(vectors)} vectors in {total_batches} batches of {BATCH_SIZE}...")

for batch_num, i in enumerate(range(0, len(vectors), BATCH_SIZE), start=1):
    batch = vectors[i:i + BATCH_SIZE]
    try:
        index.upsert(vectors=batch)
        logger.info(f"  ✓ Batch {batch_num}/{total_batches} upserted ({len(batch)} vectors)")
    except Exception as e:
        logger.error(f"  ✗ Error on batch {batch_num}: {e}")
        raise

# ─── Verify ──────────────────────────────────────────────────────
stats = index.describe_index_stats()
logger.info("=" * 60)
logger.info("Index build complete!")
logger.info(f"Total vectors in index: {stats.get('total_vector_count', 'Unknown')}")
logger.info("=" * 60)