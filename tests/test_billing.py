import pytest

from billing import calculate_bill


def test_the_900_minute_bill_the_model_got_wrong():
    bill = calculate_bill("growth", 900)
    assert bill["topups_of_300_minutes"] == 3
    assert bill["total_usd_per_month"] == 1199          # the agent once answered $999


def test_partial_topups_round_up():
    assert calculate_bill("starter", 1)["overage_usd_per_month"] == 100
    assert calculate_bill("starter", 301)["topups_of_300_minutes"] == 2


def test_no_extra_minutes():
    assert calculate_bill("scale")["total_usd_per_month"] == 1299


def test_annual_billing_applies_20_percent_off():
    assert calculate_bill("growth", billing="annual")["annual_plan_usd_per_year"] == 8630.4


@pytest.mark.parametrize("bad", [{"plan": "platinum"}, {"plan": "growth", "extra_minutes": -5},
                                 {"plan": "growth", "extra_minutes": 2.5}, {"plan": "growth", "billing": "weekly"}])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        calculate_bill(**bad)
