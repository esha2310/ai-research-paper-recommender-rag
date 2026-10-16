# AI-Powered Research Paper Recommender & RAG Assistant

![Project cover](assets/images/linkedin_cover.png)

A portfolio-focused **Machine Learning + AI** project for academic-paper retrieval,
paper selection, conservative evidence-grounded analysis, and retrieval-augmented
question answering.

> **Accuracy note:** this repository does not claim a retrieval benchmark score.
> Standard metrics are implemented, but the included evaluation files contain no
> invented relevance labels.

## Interface

![App preview](assets/images/app_preview.png)

### Core flow

```text
Research Topic / Abstract
        ↓
Find Relevant Papers
        ↓
Top Recommended Papers
        ↓
Select Paper(s)
        ↓
AI Summary / Methods / Dataset / Results / Limitations / Gap Caution
        ↓
Ask Questions with RAG
        ↓
Grounded Answer + Retrieved Sources
```


## Live demo

A self-contained browser demo is included in `docs/`.

After enabling GitHub Pages from `main` → `/docs`, the expected project URL is:

`https://esha2310.github.io/ai-research-paper-recommender-rag/`

The static demo runs browser-side TF-IDF-style retrieval and extractive evidence
retrieval. The full Python/Streamlit app remains in `app.py`.

See `LIVE_DEMO_DEPLOY.md` for deployment details.

## Features

- TF-IDF + cosine similarity retrieval baseline.
- Optional sentence-transformer semantic retrieval.
- Visible fallback if semantic mode is unavailable.
- Search by topic, keywords, or abstract.
- Source-linked recommendation cards.
- Conservative paper-analysis panels grounded in available text.
- RAG-style passage retrieval over selected papers.
- Optional user-configured external LLM endpoint.
- Extractive fallback when no LLM is configured.
- Collection analytics.
- Precision@K, Recall@K, MRR and nDCG@K implementations.
- Responsive Streamlit UI with a local original research/AI illustration.
- Tests and GitHub Actions.

## Interpretation rules

**Relevance score:** cosine similarity, not accuracy, confidence, or a calibrated probability.

**RAG output:** offline mode is extractive. External LLM use is optional and source-grounded.

**Research gap:** the app does not auto-claim one from a short abstract. Academic gap claims
should be supported by multiple full papers and their stated limitations.

## Demo data

`data/sample/papers.csv` is a small offline demonstration collection. Titles, authors,
years and URLs were checked against public source records. Its `abstract` field contains
a short **paraphrased demo summary**, not a verbatim source abstract.

For substantive use:

```bash
python scripts/fetch_arxiv.py --query "retrieval augmented generation" --max-results 100
```

## Installation

Recommended Python: **3.10–3.12**.

```bash
git clone https://github.com/esha2310/ai-research-paper-recommender-rag.git
cd ai-research-paper-recommender-rag
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

## Semantic retrieval

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The first semantic run may require internet access to download model weights. If it
cannot load, the app visibly falls back to TF-IDF.

## Optional external LLM

The project works without an API key. Copy `.env.example` to `.env` and configure
a chat-completions compatible endpoint only if you choose to use one.

```text
LLM_API_URL=
LLM_API_KEY=
LLM_MODEL=
```

Never commit `.env`.

## Build indexes

```bash
python scripts/build_index.py
python scripts/build_index.py --semantic
```

## Evaluation

The included relevance files are intentionally empty. After adding labelled queries
and qrels:

```bash
python scripts/evaluate.py --papers data/generated/arxiv_papers.csv -k 5
```

Only then should Precision@K, Recall@K, MRR or nDCG@K be reported.

## Required CSV columns

`paper_id`, `title`, `authors`, `year`, `category`, `abstract`, `paper_url`

Optional: `content_note`

## Tests

```bash
pytest -q
python -m py_compile app.py src/*.py scripts/*.py
```

## Limitations

- Offline sample is small and not a benchmark.
- Semantic retrieval requires a transformer model to be available.
- Retrieval is over metadata/abstract text unless richer text is supplied.
- Extractive fallback is not equivalent to a generative LLM.
- Cosine similarity is not a calibrated probability.
- Automated output alone does not establish a research gap.

## Author

**Nafisa Tasneem Esha**

Portfolio: https://esha2310.github.io/nafisa-tasneem-esha-portfolio/  
GitHub: https://github.com/esha2310

## Licence

MIT.
