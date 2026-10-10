"""Calldesk billing rules as plain code, so no model ever has to do pricing arithmetic.

Rules from rag/data/pricing.txt: extra minutes are sold in top-ups of 300 minutes for 100 USD;
annual billing gives 20 percent off.
"""
import math

PLANS = {
    "starter": {"price_usd_per_month": 599, "minutes": 600, "phone_numbers": 1},
    "growth": {"price_usd_per_month": 899, "minutes": 1200, "phone_numbers": 2},
    "scale": {"price_usd_per_month": 1299, "minutes": 2400, "phone_numbers": 3},
}
TOPUP_MINUTES = 300
TOPUP_PRICE_USD = 100
ANNUAL_DISCOUNT = 0.20
MAX_EXTRA_MINUTES = 100_000


def calculate_bill(plan: str, extra_minutes: int = 0, billing: str = "monthly") -> dict:
    plan = plan.strip().lower()
    if plan not in PLANS:
        raise ValueError(f"unknown plan {plan!r}; valid plans: {', '.join(PLANS)}")
    if not isinstance(extra_minutes, int) or not 0 <= extra_minutes <= MAX_EXTRA_MINUTES:
        raise ValueError(f"extra_minutes must be a whole number from 0 to {MAX_EXTRA_MINUTES}")
    if billing not in ("monthly", "annual"):
        raise ValueError("billing must be 'monthly' or 'annual'")

    base = PLANS[plan]["price_usd_per_month"]
    topups = math.ceil(extra_minutes / TOPUP_MINUTES)          # partial top-ups round up: you buy whole top-ups
    overage = topups * TOPUP_PRICE_USD
    monthly_total = base + overage
    result = {"plan": plan, "base_usd_per_month": base, "extra_minutes": extra_minutes,
              "topups_of_300_minutes": topups, "overage_usd_per_month": overage,
              "total_usd_per_month": monthly_total}
    if billing == "annual":
        result["annual_plan_usd_per_year"] = round(base * 12 * (1 - ANNUAL_DISCOUNT), 2)
        result["annual_overage_usd_per_year"] = overage * 12
        result["total_usd_per_year"] = round(result["annual_plan_usd_per_year"] + overage * 12, 2)
        result["note"] = "annual billing: 20% off the plan price (and 20% more minutes, not included here)"
    return result
