from __future__ import annotations
import math

def precision_at_k(ranked_ids, relevant, k):
    k=max(1,int(k)); return sum(x in relevant for x in ranked_ids[:k]) / k

def recall_at_k(ranked_ids, relevant, k):
    if not relevant: return 0.0
    return sum(x in relevant for x in ranked_ids[:max(1,int(k))]) / len(relevant)

def reciprocal_rank(ranked_ids, relevant):
    for rank,pid in enumerate(ranked_ids,1):
        if pid in relevant: return 1.0/rank
    return 0.0

def ndcg_at_k(ranked_ids, relevant, k):
    k=max(1,int(k))
    dcg=sum((1.0/math.log2(i+1)) for i,pid in enumerate(ranked_ids[:k],1) if pid in relevant)
    ideal=min(len(relevant),k)
    if ideal==0: return 0.0
    idcg=sum(1.0/math.log2(i+1) for i in range(1,ideal+1))
    return dcg/idcg

def evaluate_rankings(rankings, qrels, k=5):
    rows=[]
    for qid, ranked in rankings.items():
        rel=qrels.get(qid,set())
        rows.append((
            precision_at_k(ranked,rel,k), recall_at_k(ranked,rel,k),
            reciprocal_rank(ranked,rel), ndcg_at_k(ranked,rel,k)
        ))
    if not rows:
        return {"queries":0,"precision@k":None,"recall@k":None,"mrr":None,"ndcg@k":None}
    n=len(rows)
    return {
        "queries":n,
        "precision@k":sum(r[0] for r in rows)/n,
        "recall@k":sum(r[1] for r in rows)/n,
        "mrr":sum(r[2] for r in rows)/n,
        "ndcg@k":sum(r[3] for r in rows)/n,
    }
