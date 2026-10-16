from __future__ import annotations
import numpy as np
import pandas as pd

class SemanticDependencyError(RuntimeError):
    pass

class SemanticRecommender:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.df = None

    def fit(self, papers: pd.DataFrame):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise SemanticDependencyError(
                "Semantic retrieval requires sentence-transformers."
            ) from exc
        if papers.empty:
            raise ValueError("Cannot fit on an empty paper collection.")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embeddings = np.asarray(
                self.model.encode(
                    papers["search_text"].fillna("").tolist(),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            raise SemanticDependencyError(
                "The semantic model could not be loaded. It may need internet access on first use."
            ) from exc
        self.df = papers.reset_index(drop=True).copy()
        return self

    def recommend(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.model is None or self.embeddings is None or self.df is None:
            raise RuntimeError("Fit the recommender first.")
        query = str(query or "").strip()
        if not query:
            return self.df.iloc[0:0].copy().assign(relevance_score=pd.Series(dtype=float))
        top_k = max(1, min(int(top_k), len(self.df)))
        q = np.asarray(
            self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )
        scores = self.embeddings @ q
        order = np.argsort(-scores, kind="stable")[:top_k]
        result = self.df.iloc[order].copy()
        result["relevance_score"] = scores[order]
        result["retrieval_method"] = f"Sentence-transformer cosine similarity ({self.model_name})"
        return result.reset_index(drop=True)
