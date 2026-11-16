# Data Dictionary — Campaign Revenue Dataset

File: `data/campaign_revenue_raw.csv` (raw, 20,200 rows incl. ~1% duplicates)
Cleaned file produced by the pipeline: `data/campaign_revenue_clean.csv`

| Field | Type | Description | Notes |
|---|---|---|---|
| campaign_id | string | Unique campaign identifier | Used to detect/drop duplicates; not a model feature |
| campaign_start_date | date | Campaign start date | Not used as a model feature in this prototype |
| channel | categorical | Marketing channel: Social, Search, Email, Display, Affiliate | Raw data has inconsistent casing/spacing (e.g. "social ", "SOCIAL") and ~3% missing |
| planned_budget_inr | numeric | Planned campaign budget (INR) at planning stage | ~7% missing in raw data; contains injected extreme outliers |
| planned_duration_days | numeric | Planned campaign duration (days) | ~4% missing; some raw values appear as negative (sign-error, corrected in cleaning) |
| planned_audience_size | numeric | Planned audience reach | ~5% missing |
| offer_discount_pct | numeric | Planned promotional discount (%) | ~8% missing |
| revenue_inr | numeric (target) | Realized/estimated campaign revenue (INR) | Contains injected outliers (~1%); target variable |

## Engineered features (created in train_model.py / notebook)
| Field | Description |
|---|---|
| budget_per_day | planned_budget_inr / planned_duration_days |
| audience_per_budget | planned_audience_size / planned_budget_inr |

## Leakage prevention
Only planning-stage inputs are used. Realized clicks, conversions, and final spend are **not**
part of this dataset, by design, since they would only be known after the campaign runs.
