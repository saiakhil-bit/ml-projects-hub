# Data Dictionary — Lead Conversion Dataset

File: `data/lead_conversion_raw.csv` (raw, 20,200 rows incl. ~1% duplicates)
Cleaned file produced by the pipeline: `data/lead_conversion_clean.csv`

| Field | Type | Description | Notes |
|---|---|---|---|
| lead_id | string | Unique lead identifier | Used to detect/drop duplicates; not a model feature |
| lead_created_date | date | Date the lead was created | Not used as a model feature in this prototype |
| channel | categorical | Acquisition channel: Social, Search, Email, Display, Affiliate | Raw data has inconsistent casing/spacing; ~2.5% missing |
| product_interest | categorical | Product the lead is interested in: Basic Plan, Pro Plan, Enterprise Plan, Add-on | Inconsistent casing in raw data; ~2% missing |
| prior_website_visits | numeric | Number of website visits before the scoring cutoff | ~2.5% missing; contains injected extreme outliers |
| prior_email_engagement_count | numeric | Number of email engagements before the scoring cutoff | ~3% missing; contains injected invalid negative values |
| converted_30d | binary (target) | 1 if the lead converted within 30 days, else 0 | Target variable; class balance ≈ 42% positive |

## Engineered features (created in train_model.py / notebook)
| Field | Description |
|---|---|
| total_engagement | prior_website_visits + prior_email_engagement_count |
| engagement_ratio | prior_email_engagement_count / prior_website_visits (0 if visits = 0) |

## Leakage prevention
Only features measured **before** the 30-day outcome window closes are used. Every lead in
the dataset has its full 30-day outcome window observed.
