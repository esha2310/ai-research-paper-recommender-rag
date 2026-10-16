from src.evaluation import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, evaluate_rankings

def test_metric_values():
    ranked=["a","b","c"]; relevant={"b","c"}
    assert precision_at_k(ranked,relevant,2)==0.5
    assert recall_at_k(ranked,relevant,2)==0.5
    assert reciprocal_rank(ranked,relevant)==0.5
    assert 0 < ndcg_at_k(ranked,relevant,3) <= 1

def test_empty_evaluation_returns_none_metrics():
    m=evaluate_rankings({}, {}, k=5)
    assert m["queries"]==0 and m["mrr"] is None
