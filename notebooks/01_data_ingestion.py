# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data Ingestion into Delta Lake
# MAGIC **Byaaj-Bodh | Digital-Artha Track**
# MAGIC Reads `rbi_circulars.xlsx` and `lending_rates.xlsx` from the repo `data/` folder → Delta tables.

# COMMAND ----------

%pip install openpyxl pandas -q

# COMMAND ----------

import os, pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import col, trim

# Locate data/ folder — works from Repos or local
DATA_DIR = "/Workspace/Repos/<YOUR_USERNAME>/bharatbricksiim/data"
# ↑ EDIT THIS: replace <YOUR_USERNAME> with your Databricks username

assert os.path.exists(DATA_DIR),           f"❌ Not found: {DATA_DIR}"
assert os.path.exists(f"{DATA_DIR}/rbi_circulars.xlsx"),  "❌ rbi_circulars.xlsx missing"
assert os.path.exists(f"{DATA_DIR}/lending_rates.xlsx"), "❌ lending_rates.xlsx missing"
print("✅ Both Excel files found")

spark.sql("CREATE DATABASE IF NOT EXISTS byaaj_bodh_db")
spark.sql("USE byaaj_bodh_db")

# COMMAND ----------
# MAGIC %md ### Load RBI Circulars

pdf = pd.read_excel(f"{DATA_DIR}/rbi_circulars.xlsx", sheet_name="circulars",
                    engine="openpyxl", dtype=str).fillna("")
pdf.columns = [c.strip().lower().replace(" ", "_") for c in pdf.columns]

schema = StructType([StructField(c, StringType(), True) for c in pdf.columns])
df = spark.createDataFrame(pdf, schema=schema)
for c in df.columns:
    df = df.withColumn(c, trim(col(c)))

df.write.format("delta").mode("overwrite").saveAsTable("rbi_circulars")
print(f"✅ rbi_circulars: {df.count()} rows")
display(spark.sql("SELECT circular_id, clause_title, violation_type FROM rbi_circulars"))

# COMMAND ----------
# MAGIC %md ### Load Lending Rates

pdf2 = pd.read_excel(f"{DATA_DIR}/lending_rates.xlsx", sheet_name="lending_rates",
                     engine="openpyxl")
pdf2.columns = [c.strip().lower().replace(" ","_").replace("(%)","pct").replace("%","pct")
                for c in pdf2.columns]
pdf2 = pdf2.fillna("")
for c in ["min_rate_pct","median_rate_pct","max_rate_pct","regulatory_cap_pct"]:
    pdf2[c] = pd.to_numeric(pdf2[c], errors="coerce")

float_cols = ["min_rate_pct","median_rate_pct","max_rate_pct","regulatory_cap_pct"]
schema2 = StructType(
    [StructField(c, FloatType() if c in float_cols else StringType(), True)
     for c in pdf2.columns]
)
df2 = spark.createDataFrame(pdf2, schema=schema2)
df2.write.format("delta").mode("overwrite").saveAsTable("lending_rates")
print(f"✅ lending_rates: {df2.count()} rows")
display(spark.sql("SELECT lender_type, loan_category, median_rate_pct, regulatory_cap_pct FROM lending_rates"))

# COMMAND ----------
# MAGIC %md ### Verify Delta time-travel (meaningful Lakehouse usage)

display(spark.sql("DESCRIBE HISTORY rbi_circulars"))
print("✅ Notebook 01 complete — run Notebook 02 next")
