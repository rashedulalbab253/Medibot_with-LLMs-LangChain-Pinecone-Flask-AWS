import os
import re
import glob
import hashlib
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


# ───────────────────────── PDF Loading ─────────────────────────

def load_pdf_file(data_dir: str) -> list[dict]:
    """
    Load all PDF files from the given directory.
    Returns a list of dicts with 'page_content' and 'metadata'.
    Skips empty pages and logs progress.
    """
    documents = []
    pdf_files = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    pdf_files += glob.glob(os.path.join(data_dir, "*.pdf"))
    pdf_files = list(set(pdf_files))  # deduplicate

    if not pdf_files:
        logger.warning(f"No PDF files found in: {data_dir}")
        return documents

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process.")

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            num_pages = len(reader.pages)
            logger.info(f"Processing '{os.path.basename(pdf_file)}' ({num_pages} pages)...")

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    cleaned = clean_text(text)
                    if cleaned:
                        documents.append({
                            "page_content": cleaned,
                            "metadata": {
                                "source": os.path.basename(pdf_file),
                                "source_path": pdf_file,
                                "page": i + 1,
                                "total_pages": num_pages,
                            }
                        })
        except Exception as e:
            logger.error(f"Error reading '{pdf_file}': {e}")

    logger.info(f"Loaded {len(documents)} pages total from {len(pdf_files)} PDF(s).")
    return documents


def clean_text(text: str) -> str:
    """
    Clean extracted text: remove excessive whitespace, fix common OCR artifacts.
    """
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace 3+ newlines with 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove lines that are purely whitespace
    lines = [line for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    return text.strip()


# ───────────────────────── Text Splitting ─────────────────────────

def text_split(extracted_data: list[dict], chunk_size: int = 800, chunk_overlap: int = 150) -> list[dict]:
    """
    Sentence-aware text splitter with overlap.
    Splits text into chunks that respect sentence boundaries where possible.
    Also deduplicates chunks by content hash.
    """
    chunks = []
    seen_hashes = set()

    for doc in extracted_data:
        text = doc["page_content"]
        metadata = doc["metadata"]

        # Split into sentences first (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_len + sent_len > chunk_size and current_chunk:
                # Emit current chunk
                chunk_text = ' '.join(current_chunk).strip()
                _add_chunk(chunks, seen_hashes, chunk_text, metadata)

                # Overlap: keep last N characters worth of sentences
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_len = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_len += sent_len

        # Emit the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            _add_chunk(chunks, seen_hashes, chunk_text, metadata)

    logger.info(f"Created {len(chunks)} unique chunks from {len(extracted_data)} pages.")
    return chunks


def _add_chunk(chunks: list, seen_hashes: set, text: str, metadata: dict):
    """Add a chunk if it's not a duplicate and has meaningful content."""
    if len(text) < 50:  # Skip very short chunks
        return
    content_hash = hashlib.md5(text.encode()).hexdigest()
    if content_hash not in seen_hashes:
        seen_hashes.add(content_hash)
        chunks.append({
            "page_content": text,
            "metadata": {**metadata, "chunk_hash": content_hash}
        })


def filter_to_minimal_docs(docs: list[dict]) -> list[dict]:
    """
    Keep only serializable metadata fields.
    """
    minimal_docs = []
    for doc in docs:
        meta = doc["metadata"]
        minimal_docs.append({
            "page_content": doc["page_content"],
            "metadata": {
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
            }
        })
    return minimal_docs


# ───────────────────────── Embedding Model ─────────────────────────

def download_hugging_face_embeddings(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Load the SentenceTransformer embedding model.
    Returns a model that produces 384-dimensional embeddings.
    Uses caching to avoid re-downloading.
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded successfully.")
    return model


def batch_encode(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts in batches for memory efficiency.
    Returns list of embedding vectors.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(embeddings.tolist())
        if (i + batch_size) % 500 == 0:
            logger.info(f"Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts...")
    return all_embeddings