import os
import wikipedia
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from rag_utils import build_hybrid_index, hybrid_retrieve, add_to_knowledge
from ollama_utils import query_ollama  # your existing function

KNOWLEDGE_FILE = "knowledge.txt"

# Query rewriter (TF Flan-T5)

T5_MODEL = "google/flan-t5-small"
print("Loading Flan-T5 for query rewriting...")
t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
t5_model = TFAutoModelForSeq2SeqLM.from_pretrained(T5_MODEL, from_pt=True)
print("Flan-T5 loaded.")

def rewrite_query(user_input, max_length=64):
    prompt = (
        "You are an expert query rewriter. Convert the following user input into a concise, unambiguous search query suitable for retrieval.\n\n"
        f"Input: {user_input}\nOutput:"
    )
    inputs = t5_tokenizer(prompt, return_tensors="tf", truncation=True, padding="longest")
    outputs = t5_model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    rewritten = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rewritten.strip()


# Wikipedia helper

def fetch_wikipedia_summary(query, sentences=3):
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(query, sentences=sentences, auto_suggest=True, redirect=True)
    except Exception:
        return ""


# Load or build hybrid index
print("Loading hybrid index (BM25 + FAISS) from knowledge.txt...")
docs, bm25, faiss_index = build_hybrid_index(KNOWLEDGE_FILE)
print("Hybrid index ready.")

# Main loop
def main():
    global docs, bm25, faiss_index

    while True:
        q = input("\nAsk something (or type 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # Raw TinyLLaMA output
        print("\n TinyLLaMA Raw Reply:")
        raw = query_ollama(q)
        print(raw)

        # Query rewrite
        rewritten = rewrite_query(q)
        print("\n Rewritten Query:", rewritten)

        # Wikipedia fetch and append if new
        wiki_summary = fetch_wikipedia_summary(rewritten)
        if wiki_summary:
            if wiki_summary not in docs:
                add_to_knowledge(f"From Wikipedia ({rewritten}): {wiki_summary}", KNOWLEDGE_FILE)
                print("New Wikipedia info appended — rebuilding index.")
                docs, bm25, faiss_index = build_hybrid_index(KNOWLEDGE_FILE, force_rebuild=True)
            else:
                print("ℹ Wikipedia info already present; skipping append.")
        else:
            print("⚠ No Wikipedia info found.")

        # Retrieve relevant chunks
        retrieved_chunks = hybrid_retrieve(rewritten, docs, bm25, faiss_index, top_k=4)
        print("\nRetrieved (top chunks):")
        for chunk in retrieved_chunks:
            print("-", chunk[:200].replace("\n", " "))

        # Build context for RAG
        context = " ".join(retrieved_chunks)
        final_prompt = (
            f"You are an expert assistant. Use ONLY the context below to answer the user's question.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}\n\n"
            "Answer concisely. If the answer is not in the context, say 'I don't know based on the provided context.'"
        )

        # TinyLLaMA + RAG
        print("\n🔹 TinyLLaMA + RAG Reply:")
        rag_reply = query_ollama(final_prompt)
        print(rag_reply)

if __name__ == "__main__":
    main()
