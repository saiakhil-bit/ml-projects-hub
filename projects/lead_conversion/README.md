# Lead Conversion Prediction — Logistic Regression

10-day AI/ML internship prototype. Estimates the probability that a lead converts within
30 days, to help prioritize marketing follow-up.

## Business problem
Marketing teams need to prioritize follow-up on leads most likely to convert. This
prototype trains a Logistic Regression model on engagement features measured **before**
the 30-day outcome window closes, avoiding any post-outcome leakage.

## Final results (tuned model, threshold = 0.30)
| Metric | Value |
|---|---|
| Test Accuracy | 0.9997 |
| Precision | 0.9993 |
| Recall | 1.0000 |
| F1 | 0.9997 |
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9999 |
| Prevalence baseline accuracy | 0.5769 |

The tuned model (`C=1.0`, `class_weight=balanced`) vastly outperforms the prevalence
baseline. The decision threshold (0.30) was selected on validation data by maximizing F1.

## Folder structure
```
project2_lead_conversion/
├── data/
│   ├── lead_conversion_raw.csv         # raw, uncleaned (20,200 rows; missing values, outliers, dupes)
│   └── lead_conversion_clean.csv       # cleaned dataset used for modeling
├── notebooks/
│   └── lead_conversion_analysis.ipynb    # EDA + all 20 steps, narrated
├── model/
│   ├── conversion_model_pipeline.pkl   # saved sklearn Pipeline (preprocessing + model)
│   ├── model_metadata.json             # feature list, threshold, final metrics
│   └── model_comparison.csv            # LogisticRegression vs. baseline comparison table
├── assets/                             # EDA plots (distributions, class balance, ROC curve)
├── generate_raw_data.py                # script that generated the raw synthetic dataset
├── train_model.py                      # full training pipeline (Steps 1–20)
├── app.py                              # Streamlit dashboard
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
- **Single Prediction tab** — labelled inputs (channel, product interest, prior visits,
  prior email engagement), a Predict button, the conversion probability, a follow-up
  priority label (based on the validated threshold), and a live model-performance summary.
- **Batch CSV tab** — upload a CSV of leads, get probabilities + priority labels back, and
  download the results.
- Input validation: values outside the observed training range are flagged with a warning.

## Methodology summary
1. **Data cleaning**: deduplicated by `lead_id`, standardized inconsistent channel/product
   text, converted invalid negative engagement counts to missing.
2. **Missing values**: rows with missing required fields are dropped rather than imputed
   (~17,942 of 20,200 rows remain after cleaning).
3. **Outliers**: extreme `prior_website_visits` values above the 99.5th percentile removed.
4. **Feature engineering**: `total_engagement`, `engagement_ratio`.
5. **Preprocessing**: `StandardScaler` on numeric features, `OneHotEncoder` on
   `channel`/`product_interest`, combined via `ColumnTransformer` inside an `sklearn.Pipeline`.
6. **Model selection & tuning**: Logistic Regression tuned via 5-fold `GridSearchCV` over `C`
   and `class_weight`; decision threshold chosen on validation data to maximize F1.
7. **Evaluation**: precision, recall, F1, confusion matrix, ROC-AUC and PR-AUC on a held-out
   20% stratified test split, compared against a prevalence baseline. PR-AUC is reported
   alongside test prevalence since accuracy alone is insufficient for imbalanced data.

## Limitations
This is a prototype on synthetic data demonstrating the workflow. The very high separability
reflects a clear, engineered "high-intent vs. casual" engagement pattern; real-world lead
data is typically noisier, so production accuracy would be expected to be lower. Model
outputs should not be read as proof of a causal relationship between engagement and conversion.

## Troubleshooting: `AttributeError` on model load
If you see an error like `'LogisticRegression' object has no attribute 'multi_class'`
when `app.py` loads `conversion_model_pipeline.pkl`, it means the scikit-learn version in
your environment doesn't match the one used to save the pickle.

**Fix — create a clean environment matching `requirements.txt` exactly:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

**If the error persists** (pickle files are never 100% portable across every scikit-learn
build), the most reliable fix is to regenerate the model with *your own* installed
scikit-learn version rather than relying on the shipped `.pkl`:
```bash
pip install -r requirements.txt
python train_model.py   # retrains and overwrites model/conversion_model_pipeline.pkl locally
streamlit run app.py
```
