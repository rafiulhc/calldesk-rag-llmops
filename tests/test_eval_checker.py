from eval import check, gate

RESULTS = [(0.7, {"source": "pricing.txt", "chunk_id": 1, "text": "..."})]


def test_any_alternative_wording_passes():
    case = {"source": "pricing.txt", "facts": [["does not", "doesn't"]]}
    assert check(case, "No, it doesn't.", RESULTS)["answer"]


def test_missing_fact_fails():
    case = {"source": "pricing.txt", "facts": ["899"]}
    assert not check(case, "It costs 599 USD.", RESULTS)["answer"]


def test_refusal_with_curly_apostrophe_counts():
    assert check({"refuse": True}, "I don’t know based on the Calldesk docs.", [])["answer"]


def test_refusing_an_answerable_question_fails():
    case = {"source": "pricing.txt", "facts": ["899"]}
    assert not check(case, "I don't know. Maybe 899?", RESULTS)["answer"]


def test_retrieval_accepts_any_listed_source():
    case = {"source": ["product.txt", "pricing.txt"], "facts": ["x"]}
    assert check(case, "x", RESULTS)["retrieval"]


def test_eval_gate_passes_when_every_metric_reaches_the_minimum():
    assert gate({"retrieval": 100, "answer": 96, "refusal": 100}, 95) == []


def test_eval_gate_fails_and_names_the_metric_below_the_minimum():
    assert gate({"retrieval": 100, "answer": 88, "refusal": 100}, 95) == ["answer 88% < 95%"]
