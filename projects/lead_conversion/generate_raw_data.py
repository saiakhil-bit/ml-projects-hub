"""
Generates a RAW, imperfect synthetic dataset for the Lead Conversion
Prediction project (Logistic Regression).

Contains missing values, outliers, duplicates and messy category text,
by design, so cleaning / preprocessing has real work to do. The hidden
"engagement tier" driving conversion gives a strongly (but not perfectly)
separable signal, consistent with how well-qualified vs. casual leads
behave very differently in real marketing funnels.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N = 20000

CHANNELS = ["Social", "Search", "Email", "Display", "Affiliate"]
CHANNEL_RAW_VARIANTS = {
    "Social": ["Social", "social ", " SOCIAL", "Social Media"],
    "Search": ["Search", "search", "SEARCH", "Paid Search"],
    "Email": ["Email", "email", "E-mail", "EMAIL "],
    "Display": ["Display", "display", "DISPLAY"],
    "Affiliate": ["Affiliate", "affiliate", "AFFILIATE"],
}
CHANNEL_EFFECT = {"Social": 0.3, "Search": 1.1, "Email": 0.6, "Display": -0.4, "Affiliate": 0.1}

PRODUCTS = ["Basic Plan", "Pro Plan", "Enterprise Plan", "Add-on"]
PRODUCT_RAW_VARIANTS = {
    "Basic Plan": ["Basic Plan", "basic plan", "BASIC PLAN"],
    "Pro Plan": ["Pro Plan", "pro plan", "PRO PLAN"],
    "Enterprise Plan": ["Enterprise Plan", "enterprise plan", "Enterprise"],
    "Add-on": ["Add-on", "add-on", "ADD ON", "Addon"],
}
PRODUCT_EFFECT = {"Basic Plan": -0.1, "Pro Plan": 0.5, "Enterprise Plan": 0.9, "Add-on": -0.3}


def messy(mapping, key):
    return RNG.choice(mapping[key])


def main():
    lead_id = [f"LD{200000+i}" for i in range(N)]
    created_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        RNG.integers(0, 900, size=N), unit="D"
    )

    channel_true = RNG.choice(CHANNELS, size=N, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    product_true = RNG.choice(PRODUCTS, size=N, p=[0.35, 0.3, 0.2, 0.15])

    # Hidden engagement tier: ~43% of leads are genuinely "high-intent"
    tier = RNG.uniform(0, 1, size=N) < 0.42
    prior_website_visits = np.where(
        tier, RNG.poisson(35, size=N), RNG.poisson(0.5, size=N)
    ).astype(float)
    prior_email_engagement_count = np.where(
        tier, RNG.poisson(25, size=N), RNG.poisson(0.3, size=N)
    ).astype(float)

    channel_eff = np.array([CHANNEL_EFFECT[c] for c in channel_true])
    product_eff = np.array([PRODUCT_EFFECT[p] for p in product_true])

    logit = (
        -12.5
        + 0.4 * prior_website_visits
        + 0.4 * prior_email_engagement_count
        + 0.8 * channel_eff
        + 0.6 * product_eff
        + RNG.normal(0, 0.08, size=N)
    )
    prob = 1 / (1 + np.exp(-logit))
    converted_30d = (RNG.uniform(0, 1, size=N) < prob).astype(int)

    df = pd.DataFrame(
        {
            "lead_id": lead_id,
            "lead_created_date": created_dates,
            "channel": [messy(CHANNEL_RAW_VARIANTS, c) for c in channel_true],
            "product_interest": [messy(PRODUCT_RAW_VARIANTS, p) for p in product_true],
            "prior_website_visits": prior_website_visits,
            "prior_email_engagement_count": prior_email_engagement_count,
            "converted_30d": converted_30d,
        }
    )

    # ---------------- Inject REAL-WORLD messiness ----------------
    for col, frac in [
        ("channel", 0.025),
        ("product_interest", 0.02),
        ("prior_website_visits", 0.025),
        ("prior_email_engagement_count", 0.03),
    ]:
        idx = RNG.choice(N, size=int(N * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Outliers: implausible visit counts / invalid negative engagement
    out_idx = RNG.choice(N, size=int(N * 0.012), replace=False)
    df.loc[out_idx, "prior_website_visits"] = df.loc[out_idx, "prior_website_visits"] + RNG.integers(
        80, 200, size=len(out_idx)
    )
    out_idx2 = RNG.choice(N, size=int(N * 0.008), replace=False)
    df.loc[out_idx2, "prior_email_engagement_count"] = -RNG.integers(1, 10, size=len(out_idx2))

    # Duplicates (~1%)
    dup_rows = df.sample(frac=0.01, random_state=2)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sample(frac=1.0, random_state=11).reset_index(drop=True)

    df.to_csv("/home/claude/project2_lead_conversion/data/lead_conversion_raw.csv", index=False)
    print("Saved raw dataset:", df.shape)
    print("Conversion rate:", df["converted_30d"].mean())
    print(df.isna().sum())


if __name__ == "__main__":
    main()
