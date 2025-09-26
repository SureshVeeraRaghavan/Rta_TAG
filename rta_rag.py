# app.py (deployment-friendly)
import os
import io
import json
import textwrap
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# ML / embeddings / DB libs
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception as e:
    SentenceTransformer = None
    CrossEncoder = None
    print("Warning: sentence_transformers not available:", e)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception as e:
    chromadb = None
    print("Warning: chromadb not available:", e)

# Optional Google Gemini (synthesis) — don't crash if missing or key absent
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# --------- Config via env ----------
DATA_FOLDER = os.environ.get("DATA_FOLDER", "./dataset")
CSV_FILENAME = os.environ.get("CSV_FILENAME", "ridership.csv")
CSV_PATH = os.path.join(DATA_FOLDER, CSV_FILENAME)

CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "ridership_data")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # do NOT hard-code keys in source
PORT = int(os.environ.get("PORT", 10000))

# --------- LOAD DATA safely ----------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Set DATA_FOLDER/CSV_FILENAME correctly.")

df = pd.read_csv(CSV_PATH, encoding="utf-8", errors="ignore")
df.columns = df.columns.str.strip()
# try multiple parse formats gracefully
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
if df['date'].isna().any():
    # try alternative format mm/dd/YYYY
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%m/%d/%Y', errors='coerce')

if df['date'].isna().any():
    # final fallback: drop rows with bad dates (or you can raise)
    df = df[df['date'].notna()].copy()

print(f"CSV loaded. Rows: {len(df)}")

# --------- Initialize embedding model(s) (guarded) ----------
embedding_model = None
reranker = None
if SentenceTransformer is not None:
    try:
        # Consider using a smaller model in constrained envs
        embedding_model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    except Exception as e:
        print("Failed loading SentenceTransformer:", e)
        embedding_model = None

if CrossEncoder is not None:
    try:
        reranker = CrossEncoder(os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    except Exception as e:
        print("Failed loading CrossEncoder:", e)
        reranker = None

# --------- Chroma client and collection (robust) ----------
chroma_client = None
collection = None
if chromadb is not None:
    try:
        chroma_client = chromadb.Client()
        # create_collection will raise if exists — handle that
        try:
            collection = chroma_client.create_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            # fallback: try to get existing collection
            try:
                collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
            except Exception:
                collection = chroma_client.create_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        print("Warning: failed to initialize chroma client/collection:", e)
        chroma_client = None
        collection = None

# --------- Populate collection (only if empty) ----------
def populate_collection_if_empty():
    if collection is None or embedding_model is None:
        return
    try:
        # avoid re-adding on every restart
        if getattr(collection, "count", lambda: 0)() > 0:
            print("Chroma collection already populated.")
            return
    except Exception:
        # some chroma versions don't have count(); we use query to check
        try:
            q = collection.query(n_results=1)
            if q and q.get("ids"):
                print("Chroma collection appears populated.")
                return
        except Exception:
            pass

    documents, metadatas, ids = [], [], []
    transport_cols = [c for c in df.columns if c != "date" and c.lower() != "week"]
    for idx, row in df.iterrows():
        date_str = row['date'].strftime('%A, %B %d, %Y')
        ridership_details = []
        for col in transport_cols:
            val = row.get(col)
            if pd.notna(val):
                try:
                    ridership_details.append(f"{col.replace('_', ' ')} had {int(val)} trips")
                except Exception:
                    ridership_details.append(f"{col.replace('_', ' ')} had {val} trips")
        atomic_chunk = f"On {date_str}, " + "; ".join(ridership_details) + "."
        documents.append(atomic_chunk)
        metadatas.append({"row_id": int(idx), "date": row['date'].isoformat()})
        ids.append(str(idx))

    if documents:
        try:
            embeds = embedding_model.encode(documents, show_progress_bar=False)
            # ensure list-of-lists (no numpy objects)
            if hasattr(embeds, "tolist"):
                embeds = embeds.tolist()
            collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeds)
            print(f"Chroma collection created with {len(documents)} chunks.")
        except Exception as e:
            print("Failed to add documents to chroma:", e)

# call populate (optional)
populate_collection_if_empty()

# --------- Gemini / synthesis setup (optional) ----------
synthesis_enabled = False
synthesis_model = None
if genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # don't crash if model name invalid — create a safe wrapper
        synthesis_model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
        synthesis_enabled = True
    except Exception as e:
        print("Warning: Failed to configure Google generative ai:", e)
        synthesis_enabled = False

# --------- Utility functions ----------
def safe_generate_content(prompt):
    """Use Gemini if available, otherwise return the prompt's numeric summary as fallback."""
    if synthesis_enabled and synthesis_model is not None:
        try:
            resp = synthesis_model.generate_content(prompt)
            return getattr(resp, "text", str(resp))
        except Exception as e:
            print("Gemini generation failed:", e)
    # fallback: return the prompt wrapped (simple, deterministic)
    return "SYNTHESIS FALLBACK: " + textwrap.shorten(prompt, width=1200, placeholder="...")

def vector_retrieval(query, top_k=10):
    if collection is None or embedding_model is None:
        return []
    try:
        query_emb = embedding_model.encode([query], show_progress_bar=False)
        if hasattr(query_emb, "tolist"):
            query_emb = query_emb.tolist()
        results = collection.query(query_embeddings=[query_emb[0] if isinstance(query_emb[0], list) else query_emb[0]], n_results=top_k)
        docs = results.get("documents", [[]])[0] if results else []
        scores = results.get("distances", [[]])[0] if results else []
        if not docs:
            return []
        # optional reranking
        if reranker:
            rerank_pairs = [[query, doc] for doc in docs]
            try:
                rerank_scores = reranker.predict(rerank_pairs)
                reranked = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in reranked[:top_k]]
            except Exception:
                pass
        return docs[:top_k]
    except Exception as e:
        print("vector_retrieval error:", e)
        return []

# (Other helper functions from your original code such as classify_query, _slice_summary, _parse_period, _get_data_for_period...)
# For brevity in this deployment-friendly example we'll keep a simple answer route that uses vector_retrieval / fallback.

def classify_query(query: str):
    agg_keywords = ["total", "sum", "average", "mean", "max", "min", "highest", "lowest", "busiest", "least", "compare", "vs"]
    return "aggregation" if any(k in query.lower() for k in agg_keywords) else "descriptive"

def answer_query(query: str):
    ql = query.lower()
    qtype = classify_query(query)
    print("Detected query type:", qtype)

    # If aggregation -> try compute with pandas (very similar to your logic but simpler)
    if qtype == "aggregation":
        # naive: if ask for overall most trips
        if "most trips" in ql or "overall" in ql:
            transport_cols = [c for c in df.columns if c != "date" and c.lower() != "week"]
            totals = {col: int(df[col].sum()) for col in transport_cols}
            best = max(totals.items(), key=lambda x: x[1])
            context = "\n".join([f"{k}: {v}" for k, v in totals.items()])
            prompt = f"CONTEXT:\n{context}\n\nUSER QUERY:\n{query}\n\nANSWER:"
            return safe_generate_content(prompt)
        # fallback: just say we don't support the specific aggregation
        return "Aggregation requested. Please ask a specific aggregation (e.g., 'total trips for Metro in March 2024')."
    else:
        # descriptive -> use retrieval
        docs = vector_retrieval(query, top_k=10)
        if not docs:
            return "No relevant information found."
        context = "\n".join(docs)
        prompt = f"You are a ridership analysis assistant. Use ONLY the context below to answer the question in a detailed way.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
        return safe_generate_content(prompt)

@app.route("/query", methods=["POST"])
def query_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    user_query = data.get("query") or data.get("q") or ""
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    try:
        resp = answer_query(user_query)
        return jsonify({"answer": resp})
    except Exception as e:
        print("Error answering query:", e)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # ensure the population only runs at start (already called above)
    app.run(host="0.0.0.0", port=PORT)
