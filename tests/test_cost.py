import pytest

from request_log import cost_usd


def test_known_model_cost():
    # 1M input tokens x $0.15 + 1M output tokens x $0.60 = $0.75 (update if PRICES change)
    assert cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_unknown_model_returns_none_not_zero():
    assert cost_usd("gpt-banana-99", 100, 100) is None