from __future__ import annotations
import re
from typing import Iterable
import pandas as pd

def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

def prepare_papers(df: pd.DataFrame) -> pd.DataFrame:
    required = {"paper_id","title","authors","year","category","abstract","paper_url"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    for col in ["paper_id","title","authors","category","abstract","paper_url"]:
        out[col] = out[col].map(normalize_text)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out[out["paper_id"].ne("") & out["title"].ne("")].copy()
    out = out.drop_duplicates(subset=["paper_id"], keep="first").reset_index(drop=True)
    out["search_text"] = (
        out["title"].fillna("") + ". " + out["abstract"].fillna("") +
        ". Category: " + out["category"].fillna("")
    ).map(normalize_text)
    return out

def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if len(p.strip()) > 2]

def join_nonempty(values: Iterable[object], sep: str = " ") -> str:
    return sep.join(normalize_text(v) for v in values if normalize_text(v))
