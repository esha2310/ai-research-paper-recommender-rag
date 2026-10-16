from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import load_papers
from src.config import DEFAULT_SEMANTIC_MODEL

def main():
    p=argparse.ArgumentParser(description="Build reusable retrieval artefacts.")
    p.add_argument("--input",default=str(ROOT/"data"/"sample"/"papers.csv"))
    p.add_argument("--semantic",action="store_true")
    args=p.parse_args()

    df=load_papers(args.input)
    out=ROOT/"artifacts"; out.mkdir(exist_ok=True)

    vectorizer=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),sublinear_tf=True,max_features=40000)
    matrix=vectorizer.fit_transform(df["search_text"].fillna(""))
    joblib.dump(vectorizer,out/"tfidf_vectorizer.joblib")
    sparse.save_npz(out/"tfidf_matrix.npz",matrix)
    (out/"paper_ids.json").write_text(json.dumps(df["paper_id"].tolist(),indent=2),encoding="utf-8")
    print(f"Built TF-IDF index for {len(df)} records.")

    if args.semantic:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise SystemExit("Semantic build requested but sentence-transformers is unavailable.") from exc
        model=SentenceTransformer(DEFAULT_SEMANTIC_MODEL)
        emb=model.encode(df["search_text"].fillna("").tolist(),normalize_embeddings=True,show_progress_bar=True)
        np.save(out/"semantic_embeddings.npy",np.asarray(emb,dtype=np.float32))
        (out/"semantic_model.txt").write_text(DEFAULT_SEMANTIC_MODEL,encoding="utf-8")
        print(f"Built semantic embeddings with {DEFAULT_SEMANTIC_MODEL}.")

if __name__=="__main__":
    main()
