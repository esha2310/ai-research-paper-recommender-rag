from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "data" / "sample" / "papers.csv"
GENERATED_DATA = ROOT / "data" / "generated" / "arxiv_papers.csv"
ARTIFACT_DIR = ROOT / "artifacts"
HERO_IMAGE = ROOT / "assets" / "images" / "research_ai_hero.svg"

DEFAULT_SEMANTIC_MODEL = os.getenv(
    "SEMANTIC_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

LLM_API_URL = os.getenv("LLM_API_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
