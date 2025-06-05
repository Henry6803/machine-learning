import os
import faiss
import json
from sentence_transformers import SentenceTransformer

DOC_PATH = "data/rag_corpus/sample.txt"
INDEX_PATH = "data/rag_corpus/faiss.index"
DOC_MAP_PATH = "data/rag_corpus/documents.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- OFFLINE MODE ----
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def split_document(text, chunk_size=200):
    paragraphs = text.split('\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size:
            chunks.append(current.strip())
            current = ""
        current += " " + para
    if current.strip():
        chunks.append(current.strip())
    return chunks

with open(DOC_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chunks = split_document(text)

# Try loading the model strictly offline
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
except Exception as e:
    raise RuntimeError(
        f"Failed to load embedding model '{EMBEDDING_MODEL}'. "
        "Make sure it is cached locally. Error: " + str(e)
    )

embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

doc_map = {str(i): chunk for i, chunk in enumerate(chunks)}
with open(DOC_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(doc_map, f, ensure_ascii=False, indent=2)