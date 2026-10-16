from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRecommender:
    def __init__(self, max_features: int = 40000):
        self.max_features = max_features
        self.vectorizer = None
        self.matrix = None
        self.df = None

    def fit(self, papers: pd.DataFrame):
        if papers.empty:
            raise ValueError("Cannot fit on an empty paper collection.")
        self.df = papers.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1,2),
            min_df=1, max_features=self.max_features, sublinear_tf=True, norm="l2"
        )
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"].fillna(""))
        return self

    def recommend(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.vectorizer is None or self.matrix is None or self.df is None:
            raise RuntimeError("Fit the recommender first.")
        query = str(query or "").strip()
        if not query:
            return self.df.iloc[0:0].copy().assign(relevance_score=pd.Series(dtype=float))
        top_k = max(1, min(int(top_k), len(self.df)))
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        ranked = np.argsort(-scores, kind="stable")
        positive = ranked[scores[ranked] > 1e-12][:top_k]
        if len(positive) == 0:
            return self.df.iloc[0:0].copy().assign(relevance_score=pd.Series(dtype=float))
        order = positive
        result = self.df.iloc[order].copy()
        result["relevance_score"] = scores[order]
        result["retrieval_method"] = "TF-IDF cosine similarity"
        return result.reset_index(drop=True)
