from __future__ import annotations
import argparse
from pathlib import Path
import sys
import urllib.parse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import normalize_text

def fetch(query: str, max_results: int) -> pd.DataFrame:
    try:
        import feedparser
    except Exception as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    encoded = urllib.parse.quote(query)
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{encoded}&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    feed = feedparser.parse(url)
    if getattr(feed, "bozo", False) and not feed.entries:
        raise RuntimeError(f"arXiv feed error: {getattr(feed, 'bozo_exception', 'unknown error')}")

    rows=[]
    for entry in feed.entries:
        raw_id=str(entry.get("id",""))
        paper_id=raw_id.rstrip("/").split("/")[-1]
        published=str(entry.get("published",""))
        year=int(published[:4]) if len(published)>=4 and published[:4].isdigit() else None
        tags=[t.get("term","") for t in entry.get("tags",[])]
        authors=", ".join(a.get("name","") for a in entry.get("authors",[]))
        rows.append({
            "paper_id":paper_id,
            "title":normalize_text(entry.get("title","")),
            "authors":normalize_text(authors),
            "year":year,
            "category":", ".join(tags[:3]) or "Uncategorised",
            "abstract":normalize_text(entry.get("summary","")),
            "paper_url":raw_id,
            "content_note":"Fetched directly from the arXiv API; verify metadata against the source record before formal citation.",
        })
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(description="Fetch arXiv metadata for the recommender.")
    parser.add_argument("--query",required=True)
    parser.add_argument("--max-results",type=int,default=100)
    parser.add_argument("--output",default=str(ROOT/"data"/"generated"/"arxiv_papers.csv"))
    args=parser.parse_args()
    df=fetch(args.query,max(1,min(args.max_results,1000)))
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    print(f"Saved {len(df)} records to {out}")
    if df.empty:
        print("No records were returned. Try a broader query.")

if __name__=="__main__":
    main()
