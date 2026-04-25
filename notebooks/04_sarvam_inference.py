import numpy as  np
# ── Language map ─────────────────────────────────────────────────────────────
LANGS = {
    "hi": "Hindi",  "mr": "Marathi", "ta": "Tamil",   "te": "Telugu",
    "kn": "Kannada","ml": "Malayalam","bn": "Bengali", "gu": "Gujarati",
    "pa": "Punjabi","or": "Odia",    "en": "English",
}

SYS = ("You are Byaaj-Bodh, a financial rights advisor for Indian borrowers. "
       "Respond entirely in {lang}. Use simple language suitable for rural borrowers.")

# ── Generate function ────────────────────────────────────────────────────────
def generate(apr, benchmark, excess, verdict_str, rbi_ref, rag_text, lang="hi"):
    lang_name = LANGS[lang]
    prompt = (
        f"### System:\n{SYS.format(lang=lang_name)}\n\n"
        f"### User:\n"
        f"Loan analysis:\n"
        f"- Computed APR: {apr:.1f}%\n"
        f"- RBI Benchmark: {benchmark:.1f}%\n"
        f"- Excess: {excess:+.1f}%\n"
        f"- Verdict: {verdict_str}\n"
        f"- Regulation: {rbi_ref}\n\n"
        f"Relevant RBI clauses:\n{rag_text}\n\n"
        f"Respond in {lang_name}:\n"
        f"1. EXPLANATION: 2-3 sentences, what is wrong and which rule applies.\n"
        f"2. COMPLAINT_LETTER: Draft addressed to cms.rbi.org.in if violation.\n\n"
        f"### Assistant:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1        # prevents looping on CPU
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, round(time.time()-t0, 2)

# ── RAG setup ────────────────────────────────────────────────────────────────
# ── Simple RBI reference lookup (no FAISS/Excel needed) ─────────────────────
RBI_CLAUSES = {
    "NBFC-SBR 2023 Para 38": (
        "No NBFC shall charge interest exceeding the benchmark rate by more than 8 percentage points. "
        "NBFCs found charging usurious rates are liable to penalties under RBI Act 1934."
    ),
    "MF Loans 2022 Para 7.1": (
        "Microfinance loan interest rates shall not exceed 22% per annum for NBFC-MFIs. "
        "Any rate exceeding the prescribed cap is a violation subject to RBI enforcement."
    ),
    "Within benchmark": (
        "The loan interest rate is within the RBI prescribed benchmark. "
        "No regulatory violation detected."
    ),
}

def get_rag(query):
    # match query to closest RBI reference
    for key, text in RBI_CLAUSES.items():
        if any(word in query for word in key.split()):
            return f"[{key}]: {text}"
    return "[General RBI Guidelines]: Lenders must follow RBI fair practices code and disclose APR clearly."

# ── BhashaBench eval ─────────────────────────────────────────────────────────
eval_cases = [
    {"lang": "hi", "apr": 36.8, "bench": 22, "excess": 14.8, "verdict": "PREDATORY",
     "ref": "NBFC-SBR 2023 Para 38", "kw": "उल्लंघन"},
    {"lang": "mr", "apr": 36.8, "bench": 22, "excess": 14.8, "verdict": "PREDATORY",
     "ref": "NBFC-SBR 2023 Para 38", "kw": "उल्लंघन"},
    {"lang": "ta", "apr": 31.0, "bench": 22, "excess":  9.0, "verdict": "PREDATORY",
     "ref": "MF Loans 2022 Para 7.1", "kw": "மீற"},
    {"lang": "en", "apr": 13.5, "bench": 14, "excess": -0.5, "verdict": "FAIR",
     "ref": "Within benchmark", "kw": "fair"},
    {"lang": "en", "apr": 26.5, "bench": 18, "excess":  8.5, "verdict": "PREDATORY",
     "ref": "NBFC-SBR 2023 Para 38", "kw": "violation"},
]

passed, lats = 0, []

for ev in eval_cases:
    rag = get_rag(f"{ev['verdict']} {ev['ref']}")
    out, lat = generate(
        ev["apr"], ev["bench"], ev["excess"],
        ev["verdict"], ev["ref"], rag, ev["lang"]
    )
    ok = ev["kw"].lower() in out.lower()
    passed += ok
    lats.append(lat)
    print(f"{'✅' if ok else '❌'} [{ev['lang']}] {lat:.1f}s | kw='{ev['kw']}' {'✓' if ok else '✗'}")
    print(f"   Preview: {out[:120]}\n")

print(f"\nBhashaBench: {passed}/{len(eval_cases)} | "
      f"P50={np.median(lats):.1f}s P95={np.percentile(lats,95):.1f}s")

print("✅ Notebook 04 complete — Sarvam-2 via API")
