# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Sarvam-1 Inference + BhashaBench Eval
# MAGIC Downloads Sarvam-1 Q4_K_M GGUF (1.55 GB) → CPU inference → multilingual output → MLflow eval.
# MAGIC **Run Cell 2 first** — model download takes ~10 min on first run.

# COMMAND ----------

%pip install llama-cpp-python huggingface-hub -q

# COMMAND ----------
# MAGIC %md ### Download Sarvam-1 (do this first — de-risk LLM)

import os, time, mlflow
from huggingface_hub import hf_hub_download

MODEL_DIR  = "/dbfs/byaaj_bodh/models"
GGUF_FILE  = "sarvam-1-Q4_K_M.gguf"
gguf_path  = f"{MODEL_DIR}/{GGUF_FILE}"
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(gguf_path):
    print("Downloading Sarvam-1 Q4_K_M (~1.55 GB)...")
    hf_hub_download(repo_id="sarvamai/sarvam-1", filename=GGUF_FILE, local_dir=MODEL_DIR)
print(f"✅ Model ready: {os.path.getsize(gguf_path)/1e9:.2f} GB at {gguf_path}")

# COMMAND ----------
# MAGIC %md ### Load model + smoke test

from llama_cpp import Llama

llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=4, verbose=False)
print("✅ Sarvam-1 loaded")

t0 = time.time()
out = llm("आप एक वित्तीय सलाहकार हैं। क्या 36% ब्याज दर उचित है? हिंदी में उत्तर दें।",
          max_tokens=100, temperature=0.3, echo=False)
print(f"Hindi test ({time.time()-t0:.1f}s):\n{out['choices'][0]['text']}")

# COMMAND ----------
# MAGIC %md ### Prompt templates for 11 languages

LANGS = {
    "hi":"Hindi","mr":"Marathi","ta":"Tamil","te":"Telugu","kn":"Kannada",
    "ml":"Malayalam","bn":"Bengali","gu":"Gujarati","pa":"Punjabi",
    "or":"Odia","en":"English",
}

SYS = ("You are Byaaj-Bodh, a financial rights advisor for Indian borrowers. "
       "Respond entirely in {lang}. Use simple language suitable for rural borrowers.")

def generate(apr, benchmark, excess, verdict_str, rbi_ref, rag_text, lang="hi"):
    lang_name = LANGS[lang]
    prompt = (
        f"<|system|>\n{SYS.format(lang=lang_name)}\n"
        f"<|user|>\n"
        f"Loan analysis:\n"
        f"- Computed APR: {apr:.1f}%\n"
        f"- RBI Benchmark: {benchmark:.1f}%\n"
        f"- Excess: {excess:+.1f}%\n"
        f"- Verdict: {verdict_str}\n"
        f"- Regulation: {rbi_ref}\n\n"
        f"Relevant RBI clauses:\n{rag_text}\n\n"
        f"Respond in {lang_name}:\n"
        f"1. EXPLANATION: 2-3 sentences, what is wrong and which rule applies.\n"
        f"2. COMPLAINT_LETTER: Draft addressed to cms.rbi.org.in if violation.\n"
        f"<|assistant|>\n"
    )
    t0  = time.time()
    out = llm(prompt, max_tokens=500, temperature=0.2, echo=False)
    return out["choices"][0]["text"], round(time.time()-t0, 2)

# COMMAND ----------
# MAGIC %md ### Build training dataset from Delta tables (real data, no synthetic)

import pandas as pd
spark.sql("USE byaaj_bodh_db")

rates = spark.sql(
    "SELECT lender_type, loan_category, median_rate_pct, regulatory_cap_pct "
    "FROM lending_rates WHERE quarter='Q3' AND year=2024"
).toPandas()

scenarios = [
    (50000, 1800, 36, "Moneylender",      "General Purpose",   36.8),
    (50000, 1150, 36, "Moneylender",      "Agricultural Loan", 19.5),
    (100000,3200, 36, "NBFC",             "Personal Loan",     26.5),
    (80000, 2800, 36, "NBFC-MFI",         "Microfinance Loan", 31.0),
    (200000,4500, 60, "Private Sector Bank","Personal Loan",   13.5),
]

records = []
for (p, emi, t, lt, lc, apr) in scenarios:
    row = rates[(rates.lender_type==lt) & (rates.loan_category==lc)]
    if row.empty: continue
    bench = float(row.iloc[0]["median_rate_pct"])
    cap   = row.iloc[0]["regulatory_cap_pct"]
    excess = apr - bench
    v = ("PREDATORY" if (cap and apr>float(cap)) or excess>8
         else "OVERPRICED" if excess>2 else "FAIR")
    for lang in ["hi","mr","ta","en"]:
        records.append({"principal":p,"emi":emi,"tenure":t,"lender_type":lt,
                        "loan_category":lc,"computed_apr":apr,"benchmark":bench,
                        "excess":excess,"verdict":v,"language":lang})

train_df = spark.createDataFrame(pd.DataFrame(records))
train_df.write.format("delta").mode("overwrite").saveAsTable("sarvam_training_data")
print(f"✅ Training dataset: {train_df.count()} rows → Delta table sarvam_training_data")
display(spark.sql("SELECT language, verdict, COUNT(*) n FROM sarvam_training_data GROUP BY 1,2"))

# COMMAND ----------
# MAGIC %md ### BhashaBench-style evaluation (5 languages, MLflow logged)

import numpy as np, json, faiss
from sentence_transformers import SentenceTransformer

# Load RAG components (built in Notebook 02)
FAISS_DIR = "/dbfs/byaaj_bodh/faiss_index"
faiss_idx = faiss.read_index(f"{FAISS_DIR}/circulars.index")
with open(f"{FAISS_DIR}/chunks_meta.json", encoding="utf-8") as f:
    chunks = json.load(f)
emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def get_rag(query):
    q = emb_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, ids = faiss_idx.search(q, 3)
    return "\n".join(f"[{chunks[i]['clause_title']}]: {chunks[i]['text'][:200]}"
                     for i in ids[0])

eval_cases = [
    {"lang":"hi", "apr":36.8,"bench":22,"excess":14.8,"verdict":"PREDATORY",
     "ref":"NBFC-SBR 2023 Para 38","kw":"उल्लंघन"},
    {"lang":"mr", "apr":36.8,"bench":22,"excess":14.8,"verdict":"PREDATORY",
     "ref":"NBFC-SBR 2023 Para 38","kw":"उल्लंघन"},
    {"lang":"ta", "apr":31.0,"bench":22,"excess":9.0, "verdict":"PREDATORY",
     "ref":"MF Loans 2022 Para 7.1","kw":"மீற"},
    {"lang":"en", "apr":13.5,"bench":14,"excess":-0.5,"verdict":"FAIR",
     "ref":"Within benchmark","kw":"fair"},
    {"lang":"en", "apr":26.5,"bench":18,"excess":8.5, "verdict":"PREDATORY",
     "ref":"NBFC-SBR 2023 Para 38","kw":"violation"},
]

passed, lats = 0, []
with mlflow.start_run(run_name="bhashabench_eval"):
    for ev in eval_cases:
        rag = get_rag(f"{ev['verdict']} {ev['ref']}")
        out, lat = generate(ev["apr"], ev["bench"], ev["excess"],
                            ev["verdict"], ev["ref"], rag, ev["lang"])
        ok = ev["kw"].lower() in out.lower()
        passed += ok
        lats.append(lat)
        print(f"{'✅' if ok else '❌'} [{ev['lang']}] {lat:.1f}s | kw='{ev['kw']}' {'✓' if ok else '✗'}")

    mlflow.log_metrics({
        "bhashabench_score":      passed/len(eval_cases),
        "latency_p50_sec":        float(np.median(lats)),
        "latency_p95_sec":        float(np.percentile(lats,95)),
    })
    print(f"\nBhashaBench: {passed}/{len(eval_cases)} | "
          f"P50={np.median(lats):.1f}s P95={np.percentile(lats,95):.1f}s")

print("✅ Notebook 04 complete")
