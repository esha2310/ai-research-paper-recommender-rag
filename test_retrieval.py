import pandas as pd
from src.preprocessing import prepare_papers
from src.tfidf_recommender import TfidfRecommender

def tiny_papers():
    return prepare_papers(pd.DataFrame([
        {"paper_id":"a","title":"Genome transformer","authors":"X","year":2023,"category":"Bioinformatics","abstract":"Transformer models for DNA sequences.","paper_url":"u1"},
        {"paper_id":"b","title":"Image residual network","authors":"Y","year":2015,"category":"Computer Vision","abstract":"Residual networks for image recognition.","paper_url":"u2"},
        {"paper_id":"c","title":"Dense retrieval","authors":"Z","year":2020,"category":"Information Retrieval","abstract":"Dense passage retrieval for question answering.","paper_url":"u3"},
    ]))

def test_recommendation_ranks_related_document_first():
    model=TfidfRecommender().fit(tiny_papers())
    result=model.recommend("DNA genome transformer",top_k=2)
    assert result.iloc[0]["paper_id"]=="a"
    assert 1 <= len(result) <= 2
    assert result["relevance_score"].between(0,1).all()

def test_empty_query_returns_empty_result():
    assert TfidfRecommender().fit(tiny_papers()).recommend("  ",top_k=2).empty

def test_top_k_is_bounded():
    assert len(TfidfRecommender().fit(tiny_papers()).recommend("network",top_k=99)) <= 3
