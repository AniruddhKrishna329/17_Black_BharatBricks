# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — APR Computation Engine + Verdict
# MAGIC Newton-Raphson IRR, 6 edge cases, Delta-backed verdict. Unit tested + MLflow logged.

# COMMAND ----------

import mlflow
spark.sql("USE byaaj_bodh_db")

# COMMAND ----------
# MAGIC %md ### Core APR functions

def irr_annualised(cashflows, tol=1e-8, max_iter=500):
    """Newton-Raphson IRR. cashflows[0] < 0 (disbursement)."""
    r = 0.1
    for _ in range(max_iter):
        npv  = sum(cf/(1+r)**t for t, cf in enumerate(cashflows))
        d    = sum(-t*cf/(1+r)**(t+1) for t, cf in enumerate(cashflows))
        if abs(d) < 1e-12: break
        r2 = r - npv/d
        if abs(r2 - r) < tol: return r2 * 12  # monthly → annual
        r = r2
    return r * 12

def flat_to_reducing(flat_pct, n):
    """Convert flat rate to reducing balance equivalent."""
    flat_r = flat_pct / 100 / 12
    emi_flat = 1/n + flat_r       # per unit principal
    lo, hi = 0.00001, 0.08
    for _ in range(120):
        m = (lo + hi) / 2
        emi_red = m*(1+m)**n / ((1+m)**n - 1)
        hi, lo = (m, lo) if emi_red > emi_flat else (hi, m)
    return m * 12 * 100

def compute_apr(principal, emi, tenure, fee=0, insurance=0,
                prepay_pct=0, doc=0, rate_type="reducing", cashflows=None):
    """Returns dict: computed_apr_pct, edge_cases, breakdown."""
    edge, bk = [], {}

    if cashflows:                              # Edge case 4: irregular EMI
        base = irr_annualised(cashflows) * 100
        edge.append("IRREGULAR_EMI_IRR")
    elif rate_type == "flat":                  # Edge case 1: flat rate
        stated = (emi*tenure - principal) / principal * 100 / tenure * 12
        base   = flat_to_reducing(stated, tenure)
        bk["flat_gain_pct"] = round(base - stated, 2)
        edge.append("FLAT_RATE_CONVERTED")
    else:                                      # Standard reducing balance
        r = 0.01
        for _ in range(500):
            calc = principal * r*(1+r)**tenure / ((1+r)**tenure - 1)
            r += (emi - calc) / (principal * tenure) * 0.1
        base = r * 12 * 100

    # Edge cases 2,3,5,6: additive APR components
    add = {}
    if fee > 0 and fee/principal > 0.005:
        add["fee"]     = fee/principal * 12/tenure * 100;  edge.append("HIDDEN_FEE")
    if insurance > 0:
        add["ins"]     = insurance*tenure/principal * 12/tenure * 100; edge.append("BUNDLED_INS")
    if prepay_pct > 0:
        add["prepay"]  = prepay_pct/tenure * 12;           edge.append("PREPAY_PENALTY")
    if doc > 0:
        add["doc"]     = doc/principal * 12/tenure * 100;  edge.append("DOC_CHARGES")

    total = base + sum(add.values())
    return {"computed_apr_pct": round(total, 2),
            "base_apr_pct":     round(base, 2),
            "edge_cases":       edge, "breakdown": {**bk, **add}}

def get_benchmark(lender_type, loan_category):
    r = spark.sql(f"""
        SELECT median_rate_pct, regulatory_cap_pct, cap_source
        FROM lending_rates
        WHERE LOWER(lender_type)=LOWER('{lender_type}')
          AND LOWER(loan_category)=LOWER('{loan_category}')
          AND quarter='Q3' AND year=2024 LIMIT 1
    """).collect()
    if not r:
        r = spark.sql(f"""
            SELECT median_rate_pct, regulatory_cap_pct, cap_source
            FROM lending_rates WHERE LOWER(lender_type)=LOWER('{lender_type}')
            ORDER BY year DESC LIMIT 1
        """).collect()
    row = r[0] if r else None
    return {
        "median":  float(row["median_rate_pct"] or 22) if row else 22.0,
        "cap":     float(row["regulatory_cap_pct"]) if row and row["regulatory_cap_pct"] else None,
        "source":  row["cap_source"] if row else "fallback",
    }

def verdict(apr, lender_type, loan_category, hh_violation=False):
    b = get_benchmark(lender_type, loan_category)
    excess = round(apr - b["median"], 2)
    if hh_violation:
        v, ref = "PREDATORY", "MF Loans Direction 2022 Para 5.1 — 50% income cap"
    elif b["cap"] and apr > b["cap"]:
        v, ref = "PREDATORY", f"Regulatory cap {b['cap']:.1f}% exceeded ({b['source']})"
    elif excess > 8:
        v, ref = "PREDATORY", f"NBFC-SBR 2023 Para 38 — {excess:.1f}% above benchmark"
    elif excess > 2:
        v, ref = "OVERPRICED", f"NBFC-SBR 2023 Para 38 — above median, below usurious"
    else:
        v, ref = "FAIR", "Within RBI benchmark range"
    return {"verdict": v, "excess_pct": excess, "benchmark": b["median"],
            "hh_violation": hh_violation, "rbi_ref": ref}

def hh_check(emi, insurance, income, existing=0):
    burden = (emi + insurance + existing) / income * 100 if income > 0 else 0
    return {"burden_pct": round(burden, 2), "hh_violation": burden > 50,
            "clause": "MF Loans 2022 Para 5.1"}

# COMMAND ----------
# MAGIC %md ### Unit tests (5 cases, MLflow logged)

TESTS = [
    {"name":"Kamala — moneylender flat 20%→predatory",
     "inputs":{"principal":50000,"emi":1800,"tenure":36,"rate_type":"flat","fee":1500,"insurance":200},
     "income":18000, "lt":"Moneylender","lc":"General Purpose",
     "apr_range":(30,45),"expected_v":"PREDATORY"},
    {"name":"PSB agri loan — fair",
     "inputs":{"principal":100000,"emi":2860,"tenure":36},
     "income":0,"lt":"Public Sector Bank","lc":"Agricultural Loan",
     "apr_range":(8,11),"expected_v":"FAIR"},
    {"name":"MFI hidden fee + insurance — predatory",
     "inputs":{"principal":50000,"emi":1600,"tenure":36,"fee":1500,"insurance":200},
     "income":0,"lt":"NBFC-MFI","lc":"Microfinance Loan",
     "apr_range":(30,45),"expected_v":"PREDATORY"},
    {"name":"Income violation >50% burden",
     "inputs":{"principal":80000,"emi":3000,"tenure":24,"insurance":500},
     "income":5000,"lt":"NBFC","lc":"Personal Loan",
     "apr_range":(18,35),"expected_v":"PREDATORY"},
    {"name":"NBFC — overpriced not predatory",
     "inputs":{"principal":100000,"emi":2800,"tenure":48},
     "income":0,"lt":"NBFC","lc":"Personal Loan",
     "apr_range":(20,27),"expected_v":"OVERPRICED"},
]

passed = 0
with mlflow.start_run(run_name="apr_engine_tests"):
    for t in TESTS:
        r = compute_apr(**{k: v for k, v in t["inputs"].items()
                          if k in compute_apr.__code__.co_varnames})
        apr = r["computed_apr_pct"]
        hh  = hh_check(t["inputs"]["emi"], t["inputs"].get("insurance",0),
                        t["income"]) if t["income"] > 0 else {"hh_violation": False}
        v   = verdict(apr, t["lt"], t["lc"], hh["hh_violation"])
        ok  = t["apr_range"][0] <= apr <= t["apr_range"][1] and v["verdict"] == t["expected_v"]
        passed += ok
        print(f"{'✅' if ok else '❌'} {t['name']}")
        print(f"   APR={apr:.1f}% | verdict={v['verdict']} | rbi={v['rbi_ref'][:60]}")
    mlflow.log_metric("accuracy", passed/len(TESTS))
    print(f"\n{'='*40}\nAPR engine accuracy: {passed}/{len(TESTS)}")

print("✅ Notebook 03 complete")

