import pandas as pd
import pytest
from src.preprocessing import normalize_text, prepare_papers, split_sentences

def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a \n b   c ") == "a b c"

def test_prepare_papers_validates_required_columns():
    with pytest.raises(ValueError):
        prepare_papers(pd.DataFrame({"title":["x"]}))

def test_prepare_papers_deduplicates_paper_id():
    df=pd.DataFrame([
        {"paper_id":"1","title":"A","authors":"X","year":2020,"category":"NLP","abstract":"Text.","paper_url":"u"},
        {"paper_id":"1","title":"A duplicate","authors":"X","year":2020,"category":"NLP","abstract":"Text.","paper_url":"u"},
    ])
    assert len(prepare_papers(df)) == 1

def test_split_sentences():
    assert split_sentences("One sentence. Another sentence!") == ["One sentence.","Another sentence!"]
