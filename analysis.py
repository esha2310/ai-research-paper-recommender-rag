from __future__ import annotations
from collections import OrderedDict
from .preprocessing import split_sentences, normalize_text

KEYWORDS = {
    "methods": ("method","model","framework","architecture","approach","transformer","retrieval","encoder","network","attention","training","pre-train","pretrained","representation","embedding"),
    "dataset": ("dataset","benchmark","corpus","imagenet","cifar","wikipedia","pubmed","pmc","wmt","cora","citeseer","protein","data"),
    "results": ("result","outperform","achieve","improve","performance","accuracy","f1","mrr","bleu","effective","better"),
    "limitations": ("limitation","limited","challenge","however","future","shortcoming","cost","expensive","unavailable"),
}

def _extract(text, keywords, max_sentences=2):
    hits=[]
    for sent in split_sentences(text):
        low=sent.lower()
        if any(k in low for k in keywords):
            hits.append(sent)
        if len(hits) >= max_sentences:
            break
    return " ".join(hits)

def paper_analysis(row):
    abstract = normalize_text(row.get("abstract",""))
    sents = split_sentences(abstract)
    return OrderedDict([
        ("AI Summary", " ".join(sents[:2]) if sents else "No abstract or summary text is available."),
        ("Methods", _extract(abstract, KEYWORDS["methods"]) or "Not explicitly stated in the available text."),
        ("Dataset", _extract(abstract, KEYWORDS["dataset"]) or "Not explicitly stated in the available text."),
        ("Results", _extract(abstract, KEYWORDS["results"]) or "Not explicitly stated in the available text."),
        ("Limitations", _extract(abstract, KEYWORDS["limitations"]) or "Not explicitly stated in the available text."),
        ("Research Gap", "No research gap is asserted automatically. A defensible gap should be supported by multiple full sources, not inferred from a short metadata record alone."),
    ])
