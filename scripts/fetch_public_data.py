#!/usr/bin/env python3
"""
Fetch the third-party dataset used for the Tier-2 pass.

We do not redistribute someone else's data in this repository. UCI Online Retail II is
downloaded on demand and converted to parquet.

Source: UCI Machine Learning Repository, "Online Retail II"
        https://archive.ics.uci.edu/dataset/502/online+retail+ii
        1,067,371 real UK e-commerce transactions, Dec 2009 to Dec 2011.
"""
from __future__ import annotations
import sys, zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DEST = Path(__file__).resolve().parents[1] / "data"


def main():
    import urllib.request
    import pandas as pd
    DEST.mkdir(parents=True, exist_ok=True)
    zp = DEST / "online_retail_ii.zip"
    if not zp.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zp)
    with zipfile.ZipFile(zp) as z:
        name = z.namelist()[0]
        z.extract(name, DEST)
    xl = pd.ExcelFile(DEST / name)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, dtype={"Invoice": str, "StockCode": str})
        df["__sheet"] = sheet
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d.columns = [c.strip().replace(" ", "_").lower() for c in d.columns]
    for c in ("invoice", "stockcode", "description", "country", "__sheet"):
        d[c] = d[c].astype(str)
    out = DEST / "online_retail_ii.parquet"
    d.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(d):,} rows)")


if __name__ == "__main__":
    sys.exit(main())
