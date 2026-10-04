import math

from evals.report import ndcg_at, recall_at


def test_recall_at_k():
    assert recall_at(["a", "b", "c"], {"a", "z"}, k=2) == 0.5
    assert recall_at(["b", "c", "a"], {"a"}, k=2) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at(["a", "b", "x"], {"a", "b"}, k=10) == 1.0
    # one gold item at rank 2 instead of rank 1
    assert math.isclose(ndcg_at(["x", "a"], {"a"}, k=10), 1 / math.log2(3))
    assert ndcg_at(["x", "y"], {"a"}, k=10) == 0.0
