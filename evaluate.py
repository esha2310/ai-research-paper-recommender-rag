from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import load_papers
from src.evaluation import evaluate_rankings
from src.tfidf_recommender import TfidfRecommender

def main():
    p=argparse.ArgumentParser(description="Evaluate retrieval only with explicit relevance labels.")
    p.add_argument("--papers",default=str(ROOT/"data"/"sample"/"papers.csv"))
    p.add_argument("--queries",default=str(ROOT/"data"/"evaluation"/"queries.csv"))
    p.add_argument("--qrels",default=str(ROOT/"data"/"evaluation"/"qrels.csv"))
    p.add_argument("-k",type=int,default=5)
    args=p.parse_args()

    queries=pd.read_csv(args.queries)
    qrels_df=pd.read_csv(args.qrels)
    if queries.empty or qrels_df.empty:
        print(
            "Evaluation not run: no labelled query relevance data are included. "
            "Populate queries.csv and qrels.csv before reporting Precision@K, Recall@K, MRR or nDCG."
        )
        return

    papers=load_papers(args.papers)
    model=TfidfRecommender().fit(papers)
    rankings={}
    for _,row in queries.iterrows():
        rankings[str(row["query_id"])] = model.recommend(str(row["query"]),top_k=len(papers))["paper_id"].tolist()

    qrels={}
    for qid,group in qrels_df.groupby("query_id"):
        qrels[str(qid)] = set(group.loc[group["relevance"].astype(float)>0,"paper_id"].astype(str))

    print(evaluate_rankings(rankings,qrels,k=args.k))

if __name__=="__main__":
    main()
