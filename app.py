from __future__ import annotations

from html import escape
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.analysis import paper_analysis
from src.config import DEFAULT_SEMANTIC_MODEL, HERO_IMAGE, SAMPLE_DATA
from src.data_loader import load_papers, data_note
from src.llm_client import llm_configured, answer_with_configured_llm
from src.rag import extractive_answer, retrieve_passages
from src.semantic_recommender import SemanticRecommender
from src.tfidf_recommender import TfidfRecommender
from src.ui import app_css, load_svg_data_uri, paper_card_html

st.set_page_config(
    page_title="AI Research Paper Recommender & RAG Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(app_css(), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_load_sample():
    return load_papers(SAMPLE_DATA)


@st.cache_resource(show_spinner=False)
def cached_tfidf(records_json: str):
    df = pd.read_json(records_json, orient="records")
    return TfidfRecommender().fit(df)


@st.cache_resource(show_spinner=False)
def cached_semantic(records_json: str, model_name: str):
    df = pd.read_json(records_json, orient="records")
    return SemanticRecommender(model_name).fit(df)


def dataset_from_sidebar():
    st.sidebar.header("Data")
    mode = st.sidebar.radio(
        "Paper collection",
        ["Included sample", "Upload CSV"],
        help="The sample is a small offline demo. Upload a larger CSV or use the arXiv fetch script for real use.",
    )
    if mode == "Included sample":
        return cached_load_sample(), "sample"

    upload = st.sidebar.file_uploader("Upload paper metadata CSV", type=["csv"])
    if upload is None:
        st.sidebar.info("Upload a CSV to switch away from the sample collection.")
        return cached_load_sample(), "sample"

    try:
        return load_papers(upload), "uploaded"
    except Exception as exc:
        st.sidebar.error(f"CSV validation failed: {exc}")
        return cached_load_sample(), "sample"


papers, data_mode = dataset_from_sidebar()

st.sidebar.header("Retrieval")
retrieval_label = st.sidebar.selectbox(
    "Method",
    ["TF-IDF baseline", "Sentence Transformer semantic"],
)
max_k = min(12, max(3, len(papers)))
top_k = st.sidebar.slider("Number of recommendations", 3, max_k, min(5, max_k))
st.sidebar.caption(
    "The relevance score is cosine similarity, not a calibrated probability or accuracy measure."
)

left, right = st.columns([1.25, 0.75], gap="large")
with left:
    st.markdown(
        """
        <div class="hero-wrap">
          <span class="hero-kicker">Machine Learning + Semantic Search + RAG</span>
          <div class="hero-title">AI-Powered Research Paper Recommender &amp; RAG Assistant</div>
          <div class="hero-copy">
            Search a research topic or abstract, rank relevant papers, inspect source-grounded
            paper summaries, and ask questions over selected records. Retrieval scores, source
            text, and generated answers are kept separate so the interface does not overstate evidence.
          </div>
          <div style="margin-top:1rem">
            <span class="tech-pill">TF-IDF</span>
            <span class="tech-pill">Sentence Transformers</span>
            <span class="tech-pill">Cosine Similarity</span>
            <span class="tech-pill">RAG</span>
            <span class="tech-pill">Streamlit</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.image(load_svg_data_uri(HERO_IMAGE), use_container_width=True)

note = data_note(papers)
if data_mode == "sample":
    st.markdown(
        '<div class="source-note"><b>Offline demo data:</b> '
        + (note or "The included collection is for interface and smoke testing.")
        + " Use <code>scripts/fetch_arxiv.py</code> to create a live arXiv metadata file before substantive literature work.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="metric-strip">
      <div class="metric-box"><div class="label">Indexed records</div><div class="value">{len(papers):,}</div></div>
      <div class="metric-box"><div class="label">Categories</div><div class="value">{papers['category'].nunique():,}</div></div>
      <div class="metric-box"><div class="label">LLM mode</div><div class="value">{'Configured' if llm_configured() else 'Extractive fallback'}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("1. Research Topic / Abstract")

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

example_cols = st.columns(3)
examples = [
    "retrieval augmented generation and dense passage retrieval",
    "transformers for genome sequence understanding",
    "explainable deep learning for computer vision",
]
for i, text in enumerate(examples):
    if example_cols[i].button(text, use_container_width=True):
        st.session_state["query_text"] = text

query = st.text_area(
    "Enter a research topic, keywords, or an abstract",
    key="query_text",
    height=130,
    placeholder="Example: semantic retrieval and retrieval-augmented generation for scientific literature question answering",
    label_visibility="collapsed",
)

if st.button("🔎 Find Relevant Papers", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Enter a topic, keywords, or an abstract before searching.")

if query.strip():
    records_json = papers.to_json(orient="records")
    actual_method = retrieval_label

    if retrieval_label == "Sentence Transformer semantic":
        try:
            with st.spinner("Loading semantic retriever and ranking papers..."):
                model = cached_semantic(records_json, DEFAULT_SEMANTIC_MODEL)
                results = model.recommend(query, top_k=top_k)
        except Exception:
            st.warning(
                "Semantic retrieval could not be loaded in this environment, so the app is showing "
                "the TF-IDF baseline instead. No semantic result is being claimed for this run."
            )
            actual_method = "TF-IDF fallback"
            results = cached_tfidf(records_json).recommend(query, top_k=top_k)
    else:
        results = cached_tfidf(records_json).recommend(query, top_k=top_k)

    st.subheader("2. Top Recommended Papers")
    st.caption(
        f"Actual retrieval method used: {actual_method}. "
        "Scores are query–document cosine similarity within the active representation."
    )

    selected_ids = []
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        card_col, select_col = st.columns([0.88, 0.12], vertical_alignment="top")
        with card_col:
            st.markdown(paper_card_html(row, i), unsafe_allow_html=True)
        with select_col:
            if st.checkbox(
                "Select",
                key=f"select_{row['paper_id']}",
                value=i <= min(2, len(results)),
            ):
                selected_ids.append(row["paper_id"])

    selected = results[results["paper_id"].isin(selected_ids)].copy()

    st.subheader("3. Selected Paper Analysis")
    if selected.empty:
        st.info("Select at least one recommended paper to open the analysis and RAG sections.")
    else:
        for _, row in selected.iterrows():
            with st.expander(row["title"], expanded=True):
                analysis = paper_analysis(row)
                cols = st.columns(2)
                for j, (label, text) in enumerate(analysis.items()):
                    with cols[j % 2]:
                        st.markdown(
                            f'<div class="analysis-box"><b>{escape(str(label))}</b><br>{escape(str(text))}</div>',
                            unsafe_allow_html=True,
                        )

        st.subheader("4. Ask Questions with RAG")
        st.caption(
            "Offline mode retrieves supporting sentences from selected records. "
            "If a compatible external LLM endpoint is configured, the same retrieved context "
            "is sent with a strict source-grounding instruction."
        )
        question = st.text_input(
            "Question",
            placeholder="Example: What retrieval methods are described, and what evidence is available for their performance?",
        )

        if st.button("Ask selected papers", use_container_width=True) and question.strip():
            retrieved = retrieve_passages(question, selected, top_k=5)

            if llm_configured() and retrieved:
                try:
                    with st.spinner("Generating a grounded answer from retrieved context..."):
                        answer = answer_with_configured_llm(question, retrieved)
                    mode = "configured external LLM + retrieved context"
                    sources = retrieved
                except Exception:
                    st.warning(
                        "The configured LLM endpoint failed, so the app used the local extractive fallback. "
                        "No LLM answer is being claimed for this request."
                    )
                    fallback = extractive_answer(question, selected)
                    answer, sources, mode = fallback["answer"], fallback["sources"], fallback["mode"]
            else:
                fallback = extractive_answer(question, selected)
                answer, sources, mode = fallback["answer"], fallback["sources"], fallback["mode"]

            st.markdown(f'<div class="rag-answer">{escape(str(answer))}</div>', unsafe_allow_html=True)
            st.caption(f"Answer mode: {mode}")

            if sources:
                st.markdown("**Retrieved source passages**")
                for idx, src in enumerate(sources, start=1):
                    st.markdown(
                        f"**[Source {idx}] {src.title}** — similarity {src.score:.3f}\n\n"
                        f"> {src.text}"
                    )

st.divider()
st.subheader("Collection Analytics")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Paper category distribution**")
    st.bar_chart(papers["category"].value_counts())
with c2:
    st.markdown("**Publication year distribution**")
    year_counts = papers["year"].dropna().astype(int).value_counts().sort_index()
    st.bar_chart(year_counts)

with st.expander("Methodology, limitations, and responsible-use notes"):
    st.markdown(
        """
        **Retrieval:** TF-IDF is the transparent baseline. Semantic mode uses a sentence-transformer
        only when the dependency and model are actually available.

        **RAG:** The offline fallback retrieves sentences from selected records and does not pretend
        to be a generative model. An external chat-completions compatible endpoint is optional.

        **Research gaps:** The interface deliberately does not auto-assert a research gap from a short
        abstract. A defensible gap should be established through full-paper reading and multiple sources.

        **Evaluation:** The repository includes Precision@K, Recall@K, MRR and nDCG@K functions,
        but no benchmark metric is displayed unless labelled query relevance data are supplied.
        """
    )

st.caption("Built as a reproducible portfolio project by Nafisa Tasneem Esha.")
