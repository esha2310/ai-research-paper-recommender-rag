from __future__ import annotations
import pandas as pd
from .preprocessing import prepare_papers

def load_papers(source) -> pd.DataFrame:
    return prepare_papers(pd.read_csv(source))

def data_note(df: pd.DataFrame) -> str:
    if "content_note" not in df.columns:
        return ""
    notes = [str(x).strip() for x in df["content_note"].dropna().unique() if str(x).strip()]
    return " ".join(notes[:2])
