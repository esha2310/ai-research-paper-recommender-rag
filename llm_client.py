from __future__ import annotations
import requests
from .config import LLM_API_URL, LLM_API_KEY, LLM_MODEL
from .rag import context_for_llm

def llm_configured() -> bool:
    return bool(LLM_API_URL and LLM_MODEL)

def answer_with_configured_llm(question: str, retrieved):
    if not llm_configured():
        raise RuntimeError("No external LLM endpoint is configured.")
    system = (
        "You are a source-grounded academic assistant. Answer ONLY from supplied context. "
        "If context does not support a fact, say it is unavailable. Do not invent authors, datasets, "
        "results, limitations, citations, or research gaps. Cite source numbers like [Source 1]."
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":f"Question: {question}\n\nContext:\n{context_for_llm(retrieved)}"},
        ],
        "temperature": 0.1,
    }
    headers={"Content-Type":"application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    r = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data=r.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        raise RuntimeError("Unexpected response schema from configured endpoint.") from exc
