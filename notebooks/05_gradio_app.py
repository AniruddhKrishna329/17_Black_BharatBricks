# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Gradio App (Databricks App)
# MAGIC Full UI: APR computation → Delta benchmark lookup → FAISS RAG → Sarvam-1 → 11 languages.

# COMMAND ----------

%pip install gradio faiss-cpu sentence-transformers llama-cpp-python -q

# COMMAND ----------

import os, json, time, numpy as np, faiss, gradio as gr
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

spark.sql("USE byaaj_bodh_db")

FAISS_DIR = "/dbfs/byaaj_bodh/faiss_index"
MODEL_DIR = "/dbfs/byaaj_bodh/models"
GGUF_FILE = "sarvam-1-Q4_K_M.gguf"

print("Loading components...")
faiss_idx  = faiss.read_index(f"{FAISS_DIR}/circulars.index")
with open(f"{FAISS_DIR}/chunks_meta.json", encoding="utf-8") as f:
    chunks = json.load(f)
embedder   = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
llm        = Llama(model_path=f"{MODEL_DIR}/{GGUF_FILE}", n_ctx=2048, n_threads=4, verbose=False)
print("✅ All components loaded")

# COMMAND ----------
# MAGIC %md ### Pipeline functions

LANGS = {
    "Hindi (हिन्दी)":"Hindi","Marathi (मराठी)":"Marathi","Tamil (தமிழ்)":"Tamil",
    "Telugu (తెలుగు)":"Telugu","Kannada (ಕನ್ನಡ)":"Kannada","Malayalam (മലയാളം)":"Malayalam",
    "Bengali (বাংলা)":"Bengali","Gujarati (ગુજરાતી)":"Gujarati","Punjabi (ਪੰਜਾਬੀ)":"Punjabi",
    "Odia (ଓଡ଼ିଆ)":"Odia","English":"English",
}

LENDER_MAP = {
    "Nationalised / Public Sector Bank":"Public Sector Bank",
    "Private Sector Bank":"Private Sector Bank",
    "NBFC":"NBFC","NBFC-MFI (Microfinance)":"NBFC-MFI",
    "Cooperative Bank":"Cooperative Bank","Moneylender":"Moneylender",
}
CAT_MAP = {
    "Agricultural / Kisan":"Agricultural Loan","Personal / Consumer":"Personal Loan",
    "Microfinance":"Microfinance Loan","Housing":"Housing Loan",
    "MSME / Business":"MSME Loan","General Purpose":"General Purpose","Emergency":"General Purpose",
}

def get_benchmark(lt, lc):
    r = spark.sql(f"""
        SELECT median_rate_pct, regulatory_cap_pct, cap_source FROM lending_rates
        WHERE LOWER(lender_type)=LOWER('{LENDER_MAP.get(lt,lt)}')
          AND LOWER(loan_category)=LOWER('{CAT_MAP.get(lc,lc)}')
          AND quarter='Q3' AND year=2024 LIMIT 1
    """).collect()
    if r: return float(r[0][0] or 22), (float(r[0][1]) if r[0][1] else None), r[0][2]
    return 22.0, None, "fallback"

def calc_apr(p, emi, n, fee, ins):
    r = 0.01
    for _ in range(500):
        calc = p * r*(1+r)**n / ((1+r)**n - 1)
        r += (emi - calc) / (p * n) * 0.1
    return round((r*12*100) + (fee/p*12/n*100 if fee>0 else 0)
                             + (ins*n/p*12/n*100 if ins>0 else 0), 2)

def retrieve(query):
    q = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    _, ids = faiss_idx.search(q, 3)
    return "\n".join(f"[{chunks[i]['clause_title']}]: {chunks[i]['text'][:250]}"
                     for i in ids[0])

def analyse(principal, emi, tenure, lender_type, loan_cat, fee, insurance, income, language):
    lang_name = LANGS[language]
    apr       = calc_apr(principal, emi, int(tenure), fee, insurance)
    bench, cap, cap_src = get_benchmark(lender_type, loan_cat)
    excess    = round(apr - bench, 2)

    burden    = (emi+insurance)/income*100 if income > 0 else 0
    hh_viol   = burden > 50

    if hh_viol:
        verdict = "🔴 PREDATORY"; rbi_ref = "MF Loans 2022 Para 5.1 — 50% income cap exceeded"
    elif cap and apr > cap:
        verdict = "🔴 PREDATORY"; rbi_ref = f"Cap {cap:.1f}% exceeded — {cap_src}"
    elif excess > 8:
        verdict = "🔴 PREDATORY"; rbi_ref = f"NBFC-SBR 2023 Para 38 — {excess:.1f}% above benchmark"
    elif excess > 2:
        verdict = "🟡 OVERPRICED"; rbi_ref = "NBFC-SBR 2023 Para 38 — above median benchmark"
    else:
        verdict = "🟢 FAIR";       rbi_ref = "Within RBI benchmark range"

    rag = retrieve(f"{verdict} {lender_type} {loan_cat} APR violation")

    prompt = (
        f"<|system|>\nYou are Byaaj-Bodh. Respond entirely in {lang_name}. "
        f"Use simple language for rural borrowers.\n"
        f"<|user|>\n"
        f"APR={apr:.1f}% | Benchmark={bench:.1f}% | Excess={excess:+.1f}% | "
        f"Verdict={verdict}\nRepayment burden={burden:.1f}%{'  ⚠️ >50%' if hh_viol else ''}\n"
        f"Rule: {rbi_ref}\n\nRBI clauses:\n{rag}\n\n"
        f"In {lang_name}: 1. EXPLANATION (2-3 sentences)  2. COMPLAINT_LETTER to cms.rbi.org.in\n"
        f"<|assistant|>\n"
    )
    t0  = time.time()
    out = llm(prompt, max_tokens=600, temperature=0.2, echo=False)
    lat = time.time() - t0

    return (
        f"{'='*50}\nBYAAJ-BODH RESULT\n{'='*50}\n"
        f"Computed APR    : {apr:.1f}%\n"
        f"RBI Benchmark   : {bench:.1f}%\n"
        f"Excess          : {excess:+.1f}%\n"
        f"Repayment Burden: {burden:.1f}%{' ⚠️ VIOLATION' if hh_viol else ''}\n"
        f"VERDICT         : {verdict}\n"
        f"RBI Regulation  : {rbi_ref}\n"
        f"{'='*50}\n"
        f"[{lang_name} — {lat:.1f}s]\n\n"
        f"{out['choices'][0]['text']}"
    )

# COMMAND ----------
# MAGIC %md ### Launch Gradio app

with gr.Blocks(title="Byaaj-Bodh", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚖️ Byaaj-Bodh\n### India's Predatory Lending Detector")
    with gr.Row():
        with gr.Column():
            principal = gr.Number(label="Loan Principal (₹)", value=50000)
            emi       = gr.Number(label="Monthly EMI (₹)",    value=1800)
            tenure    = gr.Slider(3, 120, value=36, step=1, label="Tenure (months)")
            lender    = gr.Dropdown(list(LENDER_MAP.keys()), value="Moneylender", label="Lender Type")
            loan_cat  = gr.Dropdown(list(CAT_MAP.keys()),    value="General Purpose", label="Loan Category")
        with gr.Column():
            fee       = gr.Number(label="Processing Fee / Other Charges (₹)", value=0)
            insurance = gr.Number(label="Monthly Insurance Premium (₹)",       value=0)
            income    = gr.Number(label="Monthly Household Income (₹)",        value=0)
            language  = gr.Dropdown(list(LANGS.keys()), value="Hindi (हिन्दी)", label="Output Language")
            btn       = gr.Button("🔍 Analyse", variant="primary")
    result = gr.Textbox(label="Result", lines=25, show_copy_button=True)
    gr.Markdown("**Complaint portal:** cms.rbi.org.in | **Helpline:** 14448")
    btn.click(analyse,
              inputs=[principal,emi,tenure,lender,loan_cat,fee,insurance,income,language],
              outputs=result)

app.launch(server_name="0.0.0.0", server_port=7860, share=False)
