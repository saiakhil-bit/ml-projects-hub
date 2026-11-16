"""
Generates a RAW, imperfect synthetic dataset for the Campaign Revenue
Prediction project (Linear Regression).

The data intentionally contains:
  - Missing values in several columns
  - Outliers / implausible values
  - Duplicate rows
  - Mixed-case / inconsistent category text
so that the cleaning & preprocessing steps in the notebook have real work
to do, exactly as required by the project brief ("raw" data, not pre-cleaned).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 20000

CHANNELS = ["Social", "Search", "Email", "Display", "Affiliate"]
# True (hidden) multiplicative effect of each channel on revenue
CHANNEL_EFFECT = {
    "Social": 18000,
    "Search": 32000,
    "Email": 9000,
    "Display": 4000,
    "Affiliate": 14000,
}
# Messy raw text variants that will need cleaning in the notebook
CHANNEL_RAW_VARIANTS = {
    "Social": ["Social", "social ", " SOCIAL", "Social Media"],
    "Search": ["Search", "search", "SEARCH", "Paid Search"],
    "Email": ["Email", "email", "E-mail", "EMAIL "],
    "Display": ["Display", "display", "DISPLAY"],
    "Affiliate": ["Affiliate", "affiliate", "AFFILIATE"],
}


def messy_channel(ch):
    return RNG.choice(CHANNEL_RAW_VARIANTS[ch])


def main():
    campaign_id = [f"CMP{100000+i}" for i in range(N)]

    start_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        RNG.integers(0, 900, size=N), unit="D"
    )

    channel_true = RNG.choice(CHANNELS, size=N, p=[0.28, 0.22, 0.2, 0.18, 0.12])

    planned_budget_inr = np.round(RNG.gamma(shape=6.0, scale=15000, size=N), 2)
    planned_duration_days = RNG.integers(5, 60, size=N)
    planned_audience_size = RNG.integers(2000, 500000, size=N)
    offer_discount_pct = np.round(RNG.uniform(0, 40, size=N), 1)

    # ---- TRUE signal (kept small-noise so a linear model can recover it) ----
    base = 50000
    noise = RNG.normal(0, 9000, size=N)  # ~3% of typical revenue range -> high R2
    channel_eff = np.array([CHANNEL_EFFECT[c] for c in channel_true])

    revenue_inr = (
        base
        + 7.8 * planned_budget_inr
        + 950 * planned_duration_days
        + 0.85 * planned_audience_size
        - 480 * offer_discount_pct
        + channel_eff
        + noise
    )
    revenue_inr = np.round(np.clip(revenue_inr, 5000, None), 2)

    df = pd.DataFrame(
        {
            "campaign_id": campaign_id,
            "campaign_start_date": start_dates,
            "channel": [messy_channel(c) for c in channel_true],
            "planned_budget_inr": planned_budget_inr,
            "planned_duration_days": planned_duration_days,
            "planned_audience_size": planned_audience_size,
            "offer_discount_pct": offer_discount_pct,
            "revenue_inr": revenue_inr,
        }
    )

    # ---------------- Inject REAL-WORLD messiness ----------------
    # 1) Missing values (~6-9% in several columns)
    for col, frac in [
        ("planned_budget_inr", 0.07),
        ("planned_audience_size", 0.05),
        ("offer_discount_pct", 0.08),
        ("channel", 0.03),
        ("planned_duration_days", 0.04),
    ]:
        idx = RNG.choice(N, size=int(N * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 2) Outliers / implausible values (~2%)
    out_idx = RNG.choice(N, size=int(N * 0.015), replace=False)
    df.loc[out_idx, "planned_budget_inr"] = df.loc[out_idx, "planned_budget_inr"] * RNG.uniform(8, 15)

    out_idx2 = RNG.choice(N, size=int(N * 0.01), replace=False)
    df.loc[out_idx2, "revenue_inr"] = df.loc[out_idx2, "revenue_inr"] * RNG.uniform(5, 9)

    neg_idx = RNG.choice(N, size=int(N * 0.004), replace=False)
    df.loc[neg_idx, "planned_duration_days"] = -df.loc[neg_idx, "planned_duration_days"]

    # 3) Duplicate rows (~1%)
    dup_rows = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 4) Shuffle
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    df.to_csv("/home/claude/project1_campaign_revenue/data/campaign_revenue_raw.csv", index=False)
    print("Saved raw dataset:", df.shape)
    print(df.isna().sum())


if __name__ == "__main__":
    main()
