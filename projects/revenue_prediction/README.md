# Campaign Revenue Prediction — Linear Regression

10-day AI/ML internship prototype. Predicts estimated campaign revenue (INR) at the
**planning stage**, before a campaign runs, so marketing teams can compare budget scenarios.

## Business problem
Marketing teams need to estimate campaign revenue before committing budgets. This prototype
uses a Linear Regression model trained on planning-stage inputs only (no realized clicks,
conversions, or final spend — those would leak post-campaign information).

## Final results
| Metric | Value |
|---|---|
| Test MAE | ₹7,286 |
| Test RMSE | ₹9,125 |
| **Test R²** | **0.9990** |
| Mean-revenue baseline MAE | ₹233,033 |

The tuned model (Ridge, alpha=0.01) beats the mean-revenue baseline by a wide margin.

## Folder structure
```
project1_campaign_revenue/
├── data/
│   ├── campaign_revenue_raw.csv       # raw, uncleaned (20,200 rows; missing values, outliers, dupes)
│   └── campaign_revenue_clean.csv     # cleaned dataset used for modeling
├── notebooks/
│   └── campaign_revenue_analysis.ipynb  # EDA + all 20 steps, narrated
├── model/
│   ├── revenue_model_pipeline.pkl     # saved sklearn Pipeline (preprocessing + model)
│   ├── model_metadata.json            # feature list + final metrics
│   └── model_comparison.csv           # Linear/Ridge/Lasso/baseline comparison table
├── assets/                            # EDA plots (histograms, scatter, correlation heatmap)
├── generate_raw_data.py               # script that generated the raw synthetic dataset
├── train_model.py                     # full training pipeline (Steps 1–20)
├── app.py                             # Streamlit dashboard
├── requirements.txt
├── data_dictionary.md
└── README.md
```

## How to reproduce
```bash
pip install -r requirements.txt
python generate_raw_data.py   # (optional) regenerate the raw dataset
python train_model.py         # cleans data, trains & tunes the model, saves the pipeline
streamlit run app.py          # launch the dashboard
```

## Dashboard
The Streamlit app (`app.py`) provides:
- **Single Prediction tab** — labelled inputs (channel, budget, duration, audience size,
  discount %), a Predict button, the estimated revenue, and a live model-performance summary.
- **Batch CSV tab** — upload a CSV of campaigns, get predictions back, and download the results.
- Input validation: invalid durations are rejected; values outside the observed training
  range are flagged with a warning rather than silently extrapolated.

## Methodology summary
1. **Data cleaning**: deduplicated by `campaign_id`, standardized inconsistent channel text,
   corrected sign-error negative durations.
2. **Missing values**: rows with missing required fields are dropped rather than imputed, so
   training only sees real, observed values (~15,098 of 20,200 rows remain after cleaning).
3. **Outliers**: IQR-based outlier removal (1.5×IQR) on `planned_budget_inr` and `revenue_inr`.
4. **Feature engineering**: `budget_per_day`, `audience_per_budget`.
5. **Preprocessing**: `StandardScaler` on numeric features, `OneHotEncoder` on `channel`,
   combined via a `ColumnTransformer` inside a single `sklearn.Pipeline` (fit on training data only).
6. **Model selection & tuning**: Linear Regression, Ridge, Lasso compared; Ridge tuned via
   5-fold `GridSearchCV` over `alpha`.
7. **Evaluation**: MAE, RMSE, R² on a held-out 20% test split, compared against a mean-revenue
   baseline.

## Limitations
This is a prototype on synthetic data demonstrating the workflow. It does not establish
real-world business accuracy, and model coefficients should not be read as proof of a
causal effect of advertising spend.
