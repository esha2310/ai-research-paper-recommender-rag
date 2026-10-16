import pandas as pd
from src.preprocessing import prepare_papers
from src.rag import retrieve_passages, extractive_answer

def selected():
    return prepare_papers(pd.DataFrame([{
        "paper_id":"p1","title":"Dense Retrieval","authors":"A","year":2020,"category":"IR",
        "abstract":"The method uses a dual encoder for dense passage retrieval. It is evaluated on question answering benchmarks.",
        "paper_url":"u",
    }]))

def test_retrieve_passages_returns_sources():
    out=retrieve_passages("What method is used?",selected(),top_k=2)
    assert out and out[0].title=="Dense Retrieval"

def test_gap_question_does_not_invent_gap():
    ans=extractive_answer("What is the research gap?",selected())
    assert "not claimed automatically" in ans["answer"].lower()
