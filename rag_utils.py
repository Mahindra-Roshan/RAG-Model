import os
import json
import numpy as np
import faiss
import tensorflow_hub as hub
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize

# Config and file paths
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.json"
KNOWLEDGE_FILE = "knowledge.txt"
USE_MODULE = "https://tfhub.dev/google/universal-sentence-encoder/4"
EMB_DIM = 512  # USE output dim

# Load USE (TF Hub)
print("Loading USE (TensorFlow Hub)...")
use = hub.load(USE_MODULE)
print("USE loaded.")

# Text chunking
def semantic_chunk_text(text, max_sentences=4):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# Embedding
def embed_texts(texts, batch_size=32):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = use(batch)
        vectors.append(np.array(emb))
    return np.vstack(vectors).astype('float32')

# Build or load FAISS + BM25
def build_or_load_index(chunks, force_rebuild=False):
    # BM25
    tokenized = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(METADATA_FILE) and not force_rebuild:
        try:
            idx = faiss.read_index(FAISS_INDEX_FILE)
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if idx.ntotal == len(metadata):
                return idx, metadata, bm25
        except Exception:
            pass

    # Build index
    print("Embedding chunks and building FAISS index...")
    embeddings = embed_texts(chunks)
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(EMB_DIM)
    idx.add(embeddings)
    faiss.write_index(idx, FAISS_INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return idx, chunks, bm25

# Public wrappers
def build_hybrid_index(knowledge_file=KNOWLEDGE_FILE, force_rebuild=False):
    if not os.path.exists(knowledge_file):
        raise FileNotFoundError(f"{knowledge_file} not found. Create it first.")
    with open(knowledge_file, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = semantic_chunk_text(raw, max_sentences=4)
    index, metadata, bm25 = build_or_load_index(chunks, force_rebuild=force_rebuild)
    return metadata, bm25, index

def hybrid_score_and_retrieve(query, index, metadata, bm25, top_k=3, alpha=0.6):
    # BM25 scores
    q_tokens = query.split()
    sparse_scores = np.array(bm25.get_scores(q_tokens), dtype=float)

    # Dense scores via USE
    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, min(top_k*3, index.ntotal))
    dense_scores = np.zeros(len(metadata), dtype=float)
    for rank, idx in enumerate(I[0]):
        if idx >= 0:
            dense_scores[idx] = float(D[0][rank])

    # normalize
    if np.ptp(sparse_scores) == 0:
        sparse_norm = np.zeros_like(sparse_scores)
    else:
        sparse_norm = (sparse_scores - sparse_scores.min()) / (np.ptp(sparse_scores) + 1e-8)
    if np.ptp(dense_scores) == 0:
        dense_norm = np.zeros_like(dense_scores)
    else:
        dense_norm = (dense_scores - dense_scores.min()) / (np.ptp(dense_scores) + 1e-8)

    combined = alpha * dense_norm + (1 - alpha) * sparse_norm
    top_indices = np.argsort(combined)[::-1][:top_k]
    return [metadata[i] for i in top_indices]

def hybrid_retrieve(query, docs, bm25, faiss_index, top_k=3):
    return hybrid_score_and_retrieve(query, faiss_index, docs, bm25, top_k=top_k)

def add_to_knowledge(new_text, knowledge_file=KNOWLEDGE_FILE):
    with open(knowledge_file, "a", encoding="utf-8") as f:
        f.write("\n" + new_text.strip())
