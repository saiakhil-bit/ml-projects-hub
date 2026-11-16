"""
Campaign Revenue Prediction - full training pipeline.
Implements Steps 1-20 end to end:
 1 Data Loading | 2 Data Understanding | 3 EDA | 4 Data Cleaning
 5 Handling Missing Values | 6 Handling Outliers | 7 Feature Selection
 8 Feature Engineering | 9 Data Preprocessing | 10 Encoding Categorical Variables
 11 Feature Scaling | 12 Train-Test Splitting | 13 Model Selection
 14 Model Training | 15 Model Evaluation | 16 Hyperparameter Tuning
 17 Model Optimization | 18 Model Saving | 19 Prediction | 20 Model Deployment
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE = "/home/claude/project1_campaign_revenue"
RAW_PATH = f"{BASE}/data/campaign_revenue_raw.csv"
CLEAN_PATH = f"{BASE}/data/campaign_revenue_clean.csv"
ASSETS = f"{BASE}/assets"
MODEL_DIR = f"{BASE}/model"


# ============================================================
# STEP 1: DATA LOADING
# ============================================================
def step1_load():
    df = pd.read_csv(RAW_PATH, parse_dates=["campaign_start_date"])
    print(f"[Step 1] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================
# STEP 2: DATA UNDERSTANDING
# ============================================================
def step2_understand(df):
    print("[Step 2] Column dtypes:")
    print(df.dtypes)
    print("\n[Step 2] Summary statistics:")
    print(df.describe(include="all").T)
    print("\n[Step 2] Missing value counts:")
    print(df.isna().sum())
    return df


# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def step3_eda(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["revenue_inr"].dropna(), bins=60, ax=axes[0], color="#2b6cb0")
    axes[0].set_title("Distribution of Revenue (INR)")
    sns.scatterplot(
        data=df.sample(min(3000, len(df)), random_state=1),
        x="planned_budget_inr",
        y="revenue_inr",
        alpha=0.3,
        ax=axes[1],
        color="#2f855a",
    )
    axes[1].set_title("Planned Budget vs Revenue")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/eda_revenue_budget.png", dpi=110)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    channel_clean = df["channel"].astype(str).str.strip().str.title()
    sns.boxplot(x=channel_clean, y=df["revenue_inr"], ax=ax)
    ax.set_title("Revenue by Channel (raw text, pre-cleaning)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/eda_channel_box.png", dpi=110)
    plt.close()

    numeric_cols = ["planned_budget_inr", "planned_duration_days", "planned_audience_size", "offer_discount_pct", "revenue_inr"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/eda_corr_heatmap.png", dpi=110)
    plt.close()

    print("[Step 3] EDA plots saved to /assets (revenue distribution, budget-vs-revenue, channel boxplot, correlation heatmap)")
    print("\n[Step 3] Correlation with revenue_inr:")
    print(corr["revenue_inr"].sort_values(ascending=False))


# ============================================================
# STEP 4: DATA CLEANING
# ============================================================
def step4_clean(df):
    before = len(df)
    df = df.drop_duplicates(subset=["campaign_id"]).copy()
    print(f"[Step 4] Removed {before - len(df)} duplicate rows (by campaign_id)")

    # Standardize messy channel text: "social ", " SOCIAL", "Social Media" -> "Social"
    df["channel"] = df["channel"].astype(str).str.strip().str.title()
    df["channel"] = df["channel"].replace(
        {"Social Media": "Social", "Paid Search": "Search", "E-Mail": "Email", "Nan": np.nan}
    )

    # Fix impossible negative durations (data-entry sign errors)
    neg_mask = df["planned_duration_days"] < 0
    df.loc[neg_mask, "planned_duration_days"] = df.loc[neg_mask, "planned_duration_days"].abs()
    print(f"[Step 4] Fixed {neg_mask.sum()} negative duration values (sign error)")

    return df


# ============================================================
# STEP 5: HANDLING MISSING VALUES
# ============================================================
def step5_missing(df):
    print("[Step 5] Missing values before handling:")
    print(df.isna().sum())
    # Rows with any missing value are dropped for model training so that
    # imputed guesses never masquerade as real campaign data. With 20k
    # raw rows there is still a large, fully-observed sample to train on.
    before = len(df)
    df = df.dropna().copy()
    print(f"[Step 5] Dropped {before - len(df)} rows containing missing values -> {len(df)} rows remain")
    return df


# ============================================================
# STEP 6: HANDLING OUTLIERS
# ============================================================
def step6_outliers(df):
    before = len(df)
    for col in ["planned_budget_inr", "revenue_inr"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] >= lower) & (df[col] <= upper)
        removed = (~mask).sum()
        df = df[mask]
        print(f"[Step 6] {col}: removed {removed} IQR outliers (bounds {lower:,.0f} to {upper:,.0f})")
    print(f"[Step 6] Rows remaining after outlier removal: {len(df)} (from {before})")
    return df.reset_index(drop=True)


# ============================================================
# STEP 7: FEATURE SELECTION
# ============================================================
def step7_feature_selection(df):
    candidate_features = [
        "channel",
        "planned_budget_inr",
        "planned_duration_days",
        "planned_audience_size",
        "offer_discount_pct",
    ]
    numeric = ["planned_budget_inr", "planned_duration_days", "planned_audience_size", "offer_discount_pct"]
    corr_with_target = df[numeric + ["revenue_inr"]].corr()["revenue_inr"].drop("revenue_inr")
    print("[Step 7] Correlation of numeric candidates with revenue_inr:")
    print(corr_with_target.sort_values(ascending=False))
    print(f"[Step 7] Selected features: {candidate_features}")
    return candidate_features


# ============================================================
# STEP 8: FEATURE ENGINEERING
# ============================================================
def step8_feature_engineering(df):
    df["budget_per_day"] = df["planned_budget_inr"] / df["planned_duration_days"].replace(0, np.nan)
    df["audience_per_budget"] = df["planned_audience_size"] / df["planned_budget_inr"].replace(0, np.nan)
    df["budget_per_day"] = df["budget_per_day"].fillna(df["budget_per_day"].median())
    df["audience_per_budget"] = df["audience_per_budget"].fillna(df["audience_per_budget"].median())
    print("[Step 8] Engineered features added: budget_per_day, audience_per_budget")
    return df


# ============================================================
# STEP 9-11: PREPROCESSING / ENCODING / SCALING (built as one sklearn ColumnTransformer)
# ============================================================
def build_preprocessor(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    print(f"[Step 9-11] Preprocessing: StandardScaler on {numeric_features}, OneHotEncoder on {categorical_features}")
    return preprocessor


# ============================================================
# STEP 12: TRAIN-TEST SPLIT
# ============================================================
def step12_split(df, feature_cols, target_col="revenue_inr"):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[Step 12] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 13: MODEL SELECTION
# ============================================================
def step13_model_selection():
    print("[Step 13] Candidate models: LinearRegression, Ridge, Lasso (plus mean baseline)")
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=5000),
    }


# ============================================================
# STEP 14-15: MODEL TRAINING + EVALUATION
# ============================================================
def evaluate(pipe, X_test, y_test, name):
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"[Step 14-15] {name}: MAE={mae:,.0f} RMSE={rmse:,.0f} R2={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def step14_15_train_eval(models, preprocessor, X_train, X_test, y_train, y_test):
    results = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append(evaluate(pipe, X_test, y_test, name))

    baseline = Pipeline([("pre", preprocessor), ("model", DummyRegressor(strategy="mean"))])
    baseline.fit(X_train, y_train)
    results.append(evaluate(baseline, X_test, y_test, "MeanBaseline"))

    return pd.DataFrame(results), fitted


# ============================================================
# STEP 16-17: HYPERPARAMETER TUNING + MODEL OPTIMIZATION
# ============================================================
def step16_17_tune(preprocessor, X_train, y_train, X_test, y_test):
    pipe = Pipeline([("pre", preprocessor), ("model", Ridge())])
    param_grid = {"model__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"[Step 16-17] Best Ridge alpha: {grid.best_params_} | CV R2: {grid.best_score_:.4f}")
    best_pipe = grid.best_estimator_
    test_r2 = r2_score(y_test, best_pipe.predict(X_test))
    print(f"[Step 16-17] Tuned Ridge test R2: {test_r2:.4f}")
    return best_pipe, grid.best_params_, test_r2


# ============================================================
# STEP 18: MODEL SAVING
# ============================================================
def step18_save(final_pipe, feature_cols, metrics_row):
    joblib.dump(final_pipe, f"{MODEL_DIR}/revenue_model_pipeline.pkl")
    meta = {
        "feature_columns": feature_cols,
        "target": "revenue_inr",
        "metrics": metrics_row,
    }
    with open(f"{MODEL_DIR}/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[Step 18] Saved pipeline -> {MODEL_DIR}/revenue_model_pipeline.pkl")


# ============================================================
# STEP 19: PREDICTION (sample inference)
# ============================================================
def step19_predict(final_pipe, feature_cols):
    sample = pd.DataFrame([{
        "channel": "Search",
        "planned_budget_inr": 150000,
        "planned_duration_days": 21,
        "planned_audience_size": 120000,
        "offer_discount_pct": 10,
        "budget_per_day": 150000 / 21,
        "audience_per_budget": 120000 / 150000,
    }])[feature_cols]
    pred = final_pipe.predict(sample)[0]
    print(f"[Step 19] Sample prediction for a Search campaign: INR {pred:,.0f}")
    return pred


# ============================================================
# STEP 20: MODEL DEPLOYMENT (handled by app.py / Streamlit — see README)
# ============================================================
def step20_note():
    print("[Step 20] Deployment: load revenue_model_pipeline.pkl inside app.py (Streamlit) for on-demand predictions.")


def main():
    df = step1_load()
    df = step2_understand(df)
    step3_eda(df)
    df = step4_clean(df)
    df = step5_missing(df)
    df = step6_outliers(df)
    feature_cols_base = step7_feature_selection(df)
    df = step8_feature_engineering(df)

    numeric_features = [
        "planned_budget_inr",
        "planned_duration_days",
        "planned_audience_size",
        "offer_discount_pct",
        "budget_per_day",
        "audience_per_budget",
    ]
    categorical_features = ["channel"]
    feature_cols = numeric_features + categorical_features

    df.to_csv(CLEAN_PATH, index=False)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = step12_split(df, feature_cols)
    models = step13_model_selection()
    results_df, fitted = step14_15_train_eval(models, preprocessor, X_train, X_test, y_train, y_test)
    print("\n[Step 15] Full comparison table:\n", results_df)

    best_pipe, best_params, tuned_r2 = step16_17_tune(preprocessor, X_train, y_train, X_test, y_test)

    final_metrics = evaluate(best_pipe, X_test, y_test, "FinalTunedModel")
    step18_save(best_pipe, feature_cols, final_metrics)
    step19_predict(best_pipe, feature_cols)
    step20_note()

    results_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)
    print("\n=== DONE: Project 1 training complete ===")


if __name__ == "__main__":
    main()
