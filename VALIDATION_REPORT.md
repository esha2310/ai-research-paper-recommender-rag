# Validation Report

Project: **AI-Powered Research Paper Recommender & RAG Assistant**  
Validation date: 2026-10-17

## Checks completed

- Core unit tests: **11 passed**
- Python syntax compilation: **passed** for `app.py`, `src/*.py`, and `scripts/*.py`
- TF-IDF index build on included 10-record demo collection: **passed**
- Sample retrieval smoke test: a RAG/dense-retrieval query returned only positive-similarity matches
- Evaluation guard: **passed** — no Precision@K, Recall@K, MRR or nDCG value is reported when relevance labels are absent
- Promotional cover image manually inspected and corrected so the title/footer do not overlap
- App preview image manually inspected for text/card overlap

## Deliberately not claimed

- No retrieval benchmark score is claimed because the repository contains no fabricated relevance judgements.
- Sentence-transformer retrieval was not executed in the build environment because the model/dependency was not available locally; the app contains a visible TF-IDF fallback.
- The Streamlit server itself was not launched in the build environment because Streamlit was not installed there. The app source passed Python syntax compilation.
- The optional external LLM path was not executed because no endpoint/key was configured.
- No research gap is automatically asserted from short metadata/abstract text.

## Demo data provenance

The included sample CSV is intentionally small. Paper titles/authors/years/source links were checked against public records during project assembly. The `abstract` field contains short paraphrased summaries for offline demonstration, not verbatim source abstracts.

For substantive use, run `scripts/fetch_arxiv.py` to retrieve live arXiv metadata and verify source records before formal citation.

## Recommended first local run

```bash
python -m venv .venv
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

Then test both:
1. `TF-IDF baseline`
2. `Sentence Transformer semantic`

The first semantic run may download the configured model.
