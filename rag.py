from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import split_sentences, normalize_text

@dataclass
class RetrievedPassage:
    paper_id: str
    title: str
    text: str
    score: float

def build_passages(selected: pd.DataFrame):
    passages=[]
    for _, row in selected.iterrows():
        for sent in split_sentences(normalize_text(row.get("abstract",""))):
            if len(sent) >= 20:
                passages.append(RetrievedPassage(
                    normalize_text(row.get("paper_id","")),
                    normalize_text(row.get("title","")),
                    sent, 0.0
                ))
    return passages

def retrieve_passages(question: str, selected: pd.DataFrame, top_k: int = 5):
    question = normalize_text(question)
    passages = build_passages(selected)
    if not question or not passages:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([p.text for p in passages])
    scores = cosine_similarity(vectorizer.transform([question]), matrix).ravel()
    order = scores.argsort()[::-1][:max(1,min(top_k,len(passages)))]
    return [
        RetrievedPassage(passages[int(i)].paper_id, passages[int(i)].title, passages[int(i)].text, float(scores[int(i)]))
        for i in order
    ]

def extractive_answer(question: str, selected: pd.DataFrame, top_k: int = 4):
    retrieved = retrieve_passages(question, selected, top_k)
    q = normalize_text(question).lower()
    if not retrieved:
        return {"answer":"I could not retrieve supporting text from the selected papers.","sources":[],"mode":"extractive fallback"}
    if any(term in q for term in ["research gap","gap","novel gap"]):
        answer = (
            "A research gap is not claimed automatically from these short records. "
            "Use the retrieved passages for comparison, but establish a gap from multiple full papers and their stated limitations."
        )
    else:
        usable=[p for p in retrieved if p.score > 0]
        answer = " ".join(p.text for p in usable[:3]) if usable else (
            "The available selected text does not contain a strong lexical match for this question. "
            "Review the retrieved passages before drawing a conclusion."
        )
    return {"answer":answer,"sources":retrieved,"mode":"extractive fallback"}

def context_for_llm(retrieved):
    return "\n\n".join(f"[Source {i}] {p.title}\n{p.text}" for i,p in enumerate(retrieved,1))
