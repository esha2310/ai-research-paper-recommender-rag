from __future__ import annotations
from html import escape
from pathlib import Path
import base64

def load_svg_data_uri(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    return "data:image/svg+xml;base64," + base64.b64encode(raw.encode("utf-8")).decode("ascii")

def app_css() -> str:
    return """
<style>
:root{--ink:#15263c;--muted:#607087;--line:rgba(31,62,91,.12)}
.block-container{max-width:1280px;padding-top:1.6rem;padding-bottom:3.2rem}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
h1,h2,h3{letter-spacing:-.02em;color:var(--ink)}
.hero-wrap{border:1px solid var(--line);border-radius:28px;padding:clamp(1.25rem,3vw,2.4rem);background:radial-gradient(circle at 90% 5%,rgba(101,88,211,.17),transparent 30%),radial-gradient(circle at 5% 95%,rgba(20,125,128,.15),transparent 32%),linear-gradient(135deg,#fff 0%,#f4f9fc 100%);overflow:hidden}
.hero-kicker{display:inline-block;font-size:.78rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;padding:.46rem .7rem;border-radius:999px;color:#1c5671;background:#e9f5fa;border:1px solid rgba(28,86,113,.12)}
.hero-title{font-size:clamp(2rem,4.8vw,4rem);line-height:1.03;margin:.8rem 0 .7rem;font-weight:850;color:#13263c}
.hero-copy{font-size:clamp(1rem,1.7vw,1.16rem);color:var(--muted);line-height:1.7;max-width:66ch}
.tech-pill{display:inline-block;margin:.22rem .28rem .22rem 0;padding:.38rem .62rem;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.78);color:#35546b;font-size:.82rem;font-weight:700}
.metric-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:.9rem 0 1.3rem}
.metric-box{border:1px solid var(--line);border-radius:16px;padding:.9rem 1rem;background:rgba(255,255,255,.92)}
.metric-box .label{color:var(--muted);font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.metric-box .value{color:var(--ink);font-size:1.08rem;font-weight:800;margin-top:.2rem;overflow-wrap:anywhere}
.paper-card{border:1px solid var(--line);border-radius:18px;padding:1rem;margin:.7rem 0;background:linear-gradient(145deg,rgba(255,255,255,.98),rgba(248,251,253,.96))}
.paper-title{font-size:1.03rem;font-weight:800;color:var(--ink);line-height:1.35;overflow-wrap:anywhere}
.paper-meta{color:var(--muted);font-size:.84rem;margin:.25rem 0 .55rem;overflow-wrap:anywhere}
.score-chip{display:inline-block;padding:.28rem .5rem;border-radius:999px;font-size:.75rem;font-weight:800;color:#3f327f;background:#efedff;border:1px solid rgba(101,88,211,.14)}
.abstract{color:#43576b;font-size:.91rem;line-height:1.55;overflow-wrap:anywhere}
.source-note{border-left:4px solid #1c7891;background:#edf8fb;padding:.75rem .9rem;border-radius:10px;color:#31576a;font-size:.9rem}
.analysis-box{border:1px solid var(--line);border-radius:16px;padding:.9rem 1rem;min-height:120px;background:#fbfdff}
.rag-answer{padding:1rem 1.1rem;border-radius:16px;background:linear-gradient(135deg,#eef9f6,#f5f3ff);border:1px solid var(--line);color:#294154;line-height:1.65}
a{overflow-wrap:anywhere}
@media(max-width:780px){.metric-strip{grid-template-columns:1fr}.hero-title{font-size:2.15rem}.block-container{padding-left:.85rem;padding-right:.85rem}}
</style>
"""

def paper_card_html(row, rank: int) -> str:
    title=escape(str(row.get("title",""))); authors=escape(str(row.get("authors","")))
    year=escape(str(row.get("year",""))); category=escape(str(row.get("category","")))
    abstract=escape(str(row.get("abstract",""))); url=escape(str(row.get("paper_url","")),quote=True)
    score=float(row.get("relevance_score",0.0))
    return f"""
<div class="paper-card">
<div class="paper-title">{rank}. {title}</div>
<div class="paper-meta">{authors} · {year} · {category}</div>
<span class="score-chip">Cosine relevance: {score:.3f}</span>
<div class="abstract">{abstract}</div>
<div style="margin-top:.6rem"><a href="{url}" target="_blank">Open source record ↗</a></div>
</div>
"""
