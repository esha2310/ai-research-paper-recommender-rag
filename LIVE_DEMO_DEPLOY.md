# Live Demo Deployment

This repository has **two deployable demos**.

## Option A — GitHub Pages browser demo

The `docs/` folder contains a self-contained browser demo.

Expected URL after publishing from the repository:

`https://esha2310.github.io/ai-research-paper-recommender-rag/`

GitHub settings:
1. Open **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **main**
4. Folder: **/docs**
5. Save

This static demo uses browser-side TF-IDF-style retrieval and extractive evidence
retrieval on the included small demo collection. It does not claim sentence-transformer
or external-LLM execution.

## Option B — Full Streamlit app

Deploy `app.py` to a Python/Streamlit hosting service.

Local check:

```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

The full app supports:
- TF-IDF retrieval
- optional sentence-transformer semantic retrieval
- optional external LLM endpoint
- extractive fallback when no external LLM is configured

## Environment variables for optional LLM use

Set these in the hosting platform's secrets/environment settings, never in Git:

```text
LLM_API_URL=
LLM_API_KEY=
LLM_MODEL=
```

## Accuracy note

Do not advertise the static GitHub Pages demo as a live transformer/LLM system.
It is an interactive browser demonstration of the same user flow.
