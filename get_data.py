"""Download the CSM dataset from UCI and save it as dataset.csv next to the notebook.

The UCI archive ships an .xlsx file. The notebook was written against a CSV
exported from Excel with Italian locale settings (';' separator, ',' decimal
mark), so this script writes the same format and the notebook runs unchanged.

Dataset: CSM (Conventional and Social Media Movies) Dataset 2014 and 2015,
M. Ahmed, UCI Machine Learning Repository, https://doi.org/10.24432/C5SP5T
Licence: CC BY 4.0
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = ("https://archive.ics.uci.edu/static/public/424/"
       "csm+conventional+and+social+media+movies+dataset+2014+and+2015.zip")
OUT = Path(__file__).resolve().parent / "dataset.csv"

with urllib.request.urlopen(URL) as resp:
    archive = zipfile.ZipFile(io.BytesIO(resp.read()))

xlsx_name = next(n for n in archive.namelist() if n.endswith(".xlsx"))
df = pd.read_excel(archive.open(xlsx_name), engine="openpyxl")

# The sheet ends with an empty row, which turns every numeric column into
# float. Drop it and restore int for complete whole-number columns: the
# notebook binary-encodes Genre, which only works on ints.
df = df.dropna(how="all")
for col in df.select_dtypes("number"):
    if df[col].notna().all() and (df[col] % 1 == 0).all():
        df[col] = df[col].astype("int64")

df.to_csv(OUT, sep=";", decimal=",", index=False)
print(f"Saved {len(df)} rows x {len(df.columns)} columns to {OUT}")
