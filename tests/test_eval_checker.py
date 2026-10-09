from eval import check

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
