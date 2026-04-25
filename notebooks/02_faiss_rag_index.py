# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — FAISS RAG Index
# MAGIC Chunks RBI circulars from Delta → embeds with multilingual MiniLM → FAISS index on DBFS.

# COMMAND ----------

%pip install faiss-cpu sentence-transformers -q

# COMMAND ----------

import os, json, numpy as np, faiss, mlflow
from sentence_transformers import SentenceTransformer

spark.sql("USE byaaj_bodh_db")
FAISS_DIR = "/dbfs/byaaj_bodh/faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)

# COMMAND ----------
# MAGIC %md ### Chunk circulars

def chunk(text, size=1600, overlap=200):
    chunks, s = [], 0
    while s < len(text):
        e = min(s + size, len(text))
        c = text[s:e].strip()
        if len(c) > 50: chunks.append(c)
        if e == len(text): break
        s += size - overlap
    return chunks

rows = spark.sql("SELECT * FROM rbi_circulars").toPandas()
all_chunks = []
for _, r in rows.iterrows():
    for i, c in enumerate(chunk(r["full_text"])):
        all_chunks.append({
            "chunk_id":       f"{r['circular_id']}_c{i:02d}",
            "circular_id":    r["circular_id"],
            "clause_title":   r["clause_title"],
            "violation_type": r["violation_type"],
            "text":           c,
        })
print(f"✅ {len(all_chunks)} chunks from {len(rows)} clauses")

# COMMAND ----------
# MAGIC %md ### Embed → FAISS

print("Loading paraphrase-multilingual-MiniLM-L12-v2 (CPU, ~300 MB download on first run)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode([c["text"] for c in all_chunks],
                          batch_size=32, show_progress_bar=True,
                          convert_to_numpy=True)

idx = faiss.IndexFlatL2(embeddings.shape[1])
idx.add(embeddings.astype(np.float32))
faiss.write_index(idx, f"{FAISS_DIR}/circulars.index")

with open(f"{FAISS_DIR}/chunks_meta.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False)

print(f"✅ FAISS index: {idx.ntotal} vectors, dim={embeddings.shape[1]}")

# COMMAND ----------
# MAGIC %md ### Sanity check + MLflow precision@1

def retrieve(query, k=3):
    q = model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, ids = idx.search(q, k)
    return [all_chunks[i] for i in ids[0]]

checks = [
    ("50% repayment burden microfinance cap", "MFL-001"),
    ("usurious interest rate NBFC penalty",   "NBFC-SBR-002"),
    ("processing fee hidden APR disclosure",  "NBFC-SBR-003"),
    ("RBI Ombudsman complaint portal",        "MFL-007"),
]

with mlflow.start_run(run_name="faiss_index"):
    mlflow.log_params({"chunks": len(all_chunks), "dim": embeddings.shape[1],
                       "model": "multilingual-MiniLM-L12-v2"})
    correct = sum(retrieve(q,1)[0]["circular_id"] == cid for q, cid in checks)
    mlflow.log_metric("retrieval_precision_at_1", correct / len(checks))
    print(f"Retrieval precision@1: {correct}/{len(checks)}")

print("✅ Notebook 02 complete")
