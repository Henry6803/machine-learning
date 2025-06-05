import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INDEX_PATH = "data/rag_corpus/faiss.index"
DOC_MAP_PATH = "data/rag_corpus/documents.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Always force offline mode for Hugging Face

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

try:
    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=CACHE_DIR, local_files_only=True)
except Exception as e:
    raise RuntimeError(
        f"Failed to load embedding model '{EMBEDDING_MODEL}'. "
        f"Make sure it is cached locally in {CACHE_DIR}. Error: {e}"
    )

# Load FAISS index and document mapping
try:
    index = faiss.read_index(INDEX_PATH)
except Exception as e:
    raise RuntimeError(
        f"Failed to load FAISS index from '{INDEX_PATH}'. Error: {e}"
    )

try:
    with open(DOC_MAP_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
except Exception as e:
    raise RuntimeError(
        f"Failed to load documents from '{DOC_MAP_PATH}'. Error: {e}"
    )

def retrieve_context(query, top_k=3):
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype('float32'), top_k)
    results = [docs[str(idx)] for idx in I[0] if str(idx) in docs]
    return "\n".join(results) if results else "No relevant documents found."