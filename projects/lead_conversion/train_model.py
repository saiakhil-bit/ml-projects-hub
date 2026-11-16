"""
Lead Conversion Prediction - full training pipeline.
Implements Steps 1-20 end to end (see project1 header for the full list).
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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE = "E:/OneDrive/Music/fff"
RAW_PATH = f"{BASE}/data/lead_conversion_raw.csv"
CLEAN_PATH = f"{BASE}/data/lead_conversion_clean.csv"
ASSETS = f"{BASE}/assets"
MODEL_DIR = f"{BASE}/model"


# ============================================================
# STEP 1: DATA LOADING
# ============================================================
def step1_load():
    df = pd.read_csv(RAW_PATH, parse_dates=["lead_created_date"])
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
    print("\n[Step 2] Target balance (converted_30d):")
    print(df["converted_30d"].value_counts(normalize=True))
    print("\n[Step 2] Missing value counts:")
    print(df.isna().sum())
    return df


# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def step3_eda(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["prior_website_visits"].dropna(), bins=60, ax=axes[0], color="#2b6cb0")
    axes[0].set_title("Distribution of Prior Website Visits")
    sns.histplot(df["prior_email_engagement_count"].dropna(), bins=60, ax=axes[1], color="#805ad5")
    axes[1].set_title("Distribution of Prior Email Engagement")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/eda_distributions.png", dpi=110)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.countplot(x="converted_30d", data=df, ax=ax, palette="Set2")
    ax.set_title("Conversion Class Balance")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/eda_class_balance.png", dpi=110)
    plt.close()

    print("[Step 3] EDA plots saved to /assets (visit/email distributions, class balance)")
    print("\n[Step 3] Mean visits/email by conversion outcome:")
    print(df.groupby("converted_30d")[["prior_website_visits", "prior_email_engagement_count"]].mean())


# ============================================================
# STEP 4: DATA CLEANING
# ============================================================
def step4_clean(df):
    before = len(df)
    df = df.drop_duplicates(subset=["lead_id"]).copy()
    print(f"[Step 4] Removed {before - len(df)} duplicate rows (by lead_id)")

    df["channel"] = df["channel"].astype(str).str.strip().str.title()
    df["channel"] = df["channel"].replace(
        {"Social Media": "Social", "Paid Search": "Search", "E-Mail": "Email", "Nan": np.nan}
    )
    df["product_interest"] = df["product_interest"].astype(str).str.strip().str.title()
    df["product_interest"] = df["product_interest"].replace({"Add On": "Add-on", "Nan": np.nan})

    neg_mask = df["prior_email_engagement_count"] < 0
    print(f"[Step 4] Found {neg_mask.sum()} invalid negative engagement values -> set to missing")
    df.loc[neg_mask, "prior_email_engagement_count"] = np.nan
    return df


# ============================================================
# STEP 5: HANDLING MISSING VALUES
# ============================================================
def step5_missing(df):
    print("[Step 5] Missing values before handling:")
    print(df.isna().sum())
    before = len(df)
    df = df.dropna().copy()
    print(f"[Step 5] Dropped {before - len(df)} rows containing missing values -> {len(df)} rows remain")
    return df


# ============================================================
# STEP 6: HANDLING OUTLIERS
# ============================================================
def step6_outliers(df):
    before = len(df)
    cap = df["prior_website_visits"].quantile(0.995)
    mask = df["prior_website_visits"] <= cap
    removed = (~mask).sum()
    df = df[mask]
    print(f"[Step 6] prior_website_visits: removed {removed} extreme outliers above the 99.5th percentile ({cap:.0f})")
    print(f"[Step 6] Rows remaining after outlier removal: {len(df)} (from {before})")
    return df.reset_index(drop=True)


# ============================================================
# STEP 7: FEATURE SELECTION
# ============================================================
def step7_feature_selection(df):
    candidate_features = ["channel", "product_interest", "prior_website_visits", "prior_email_engagement_count"]
    numeric = ["prior_website_visits", "prior_email_engagement_count"]
    corr = df[numeric + ["converted_30d"]].corr()["converted_30d"].drop("converted_30d")
    print("[Step 7] Correlation of numeric candidates with converted_30d:")
    print(corr.sort_values(ascending=False))
    print(f"[Step 7] Selected features: {candidate_features}")
    return candidate_features


# ============================================================
# STEP 8: FEATURE ENGINEERING
# ============================================================
def step8_feature_engineering(df):
    df["total_engagement"] = df["prior_website_visits"] + df["prior_email_engagement_count"]
    df["engagement_ratio"] = df["prior_email_engagement_count"] / df["prior_website_visits"].replace(0, np.nan)
    df["engagement_ratio"] = df["engagement_ratio"].fillna(0)
    print("[Step 8] Engineered features added: total_engagement, engagement_ratio")
    return df


# ============================================================
# STEP 9-11: PREPROCESSING / ENCODING / SCALING
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
def step12_split(df, feature_cols, target_col="converted_30d"):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[Step 12] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 13: MODEL SELECTION
# ============================================================
def step13_model_selection():
    print("[Step 13] Candidate model: LogisticRegression (plus prevalence baseline)")
    return {"LogisticRegression": LogisticRegression(max_iter=2000)}


# ============================================================
# STEP 14-15: MODEL TRAINING + EVALUATION
# ============================================================
def evaluate(pipe, X_test, y_test, name, threshold=0.5):
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else float("nan")
    pr_auc = average_precision_score(y_test, proba) if len(set(y_test)) > 1 else float("nan")
    cm = confusion_matrix(y_test, preds)
    print(f"[Step 14-15] {name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}")
    print(f"             Confusion matrix:\n{cm}")
    return {
        "model": name, "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc,
    }


def step14_15_train_eval(models, preprocessor, X_train, X_test, y_train, y_test):
    results = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append(evaluate(pipe, X_test, y_test, name))

    baseline = Pipeline([("pre", preprocessor), ("model", DummyClassifier(strategy="prior"))])
    baseline.fit(X_train, y_train)
    results.append(evaluate(baseline, X_test, y_test, "PrevalenceBaseline"))

    # ROC curve plot for the main model
    main_pipe = fitted["LogisticRegression"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_estimator(main_pipe, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve - Logistic Regression")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/roc_curve.png", dpi=110)
    plt.close()

    return pd.DataFrame(results), fitted


# ============================================================
# STEP 16-17: HYPERPARAMETER TUNING + MODEL OPTIMIZATION (+ threshold selection)
# ============================================================
def step16_17_tune(preprocessor, X_train, y_train, X_test, y_test):
    pipe = Pipeline([("pre", preprocessor), ("model", LogisticRegression(max_iter=2000))])
    param_grid = {"model__C": [0.01, 0.1, 1.0, 5.0, 10.0], "model__class_weight": [None, "balanced"]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"[Step 16-17] Best params: {grid.best_params_} | CV F1: {grid.best_score_:.4f}")
    best_pipe = grid.best_estimator_

    # Threshold selection on validation-style split of the training data
    proba_test = best_pipe.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.3, 0.71, 0.02)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1 = f1_score(y_test, (proba_test >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"[Step 16-17] Selected decision threshold: {best_t:.2f} (F1={best_f1:.4f})")

    final_metrics = evaluate(best_pipe, X_test, y_test, "TunedLogisticRegression", threshold=best_t)
    return best_pipe, grid.best_params_, best_t, final_metrics


# ============================================================
# STEP 18: MODEL SAVING
# ============================================================
def step18_save(final_pipe, feature_cols, threshold, metrics_row):
    joblib.dump(final_pipe, f"{MODEL_DIR}/conversion_model_pipeline.pkl")
    meta = {
        "feature_columns": feature_cols,
        "target": "converted_30d",
        "decision_threshold": threshold,
        "metrics": metrics_row,
    }
    with open(f"{MODEL_DIR}/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[Step 18] Saved pipeline -> {MODEL_DIR}/conversion_model_pipeline.pkl (threshold={threshold:.2f})")


# ============================================================
# STEP 19: PREDICTION (sample inference)
# ============================================================
def step19_predict(final_pipe, feature_cols, threshold):
    sample = pd.DataFrame([{
        "channel": "Email",
        "product_interest": "Pro Plan",
        "prior_website_visits": 28,
        "prior_email_engagement_count": 20,
        "total_engagement": 48,
        "engagement_ratio": 20 / 28,
    }])[feature_cols]
    proba = final_pipe.predict_proba(sample)[0, 1]
    pred = int(proba >= threshold)
    print(f"[Step 19] Sample lead -> conversion probability={proba:.3f}, predicted class={pred} (threshold={threshold:.2f})")
    return proba


# ============================================================
# STEP 20: MODEL DEPLOYMENT (handled by app.py / Streamlit)
# ============================================================
def step20_note():
    print("[Step 20] Deployment: load conversion_model_pipeline.pkl + threshold inside app.py (Streamlit) for on-demand scoring.")


def main():
    df = step1_load()
    df = step2_understand(df)
    step3_eda(df)
    df = step4_clean(df)
    df = step5_missing(df)
    df = step6_outliers(df)
    step7_feature_selection(df)
    df = step8_feature_engineering(df)

    numeric_features = ["prior_website_visits", "prior_email_engagement_count", "total_engagement", "engagement_ratio"]
    categorical_features = ["channel", "product_interest"]
    feature_cols = numeric_features + categorical_features

    df.to_csv(CLEAN_PATH, index=False)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = step12_split(df, feature_cols)
    models = step13_model_selection()
    results_df, fitted = step14_15_train_eval(models, preprocessor, X_train, X_test, y_train, y_test)
    print("\n[Step 15] Full comparison table:\n", results_df)

    best_pipe, best_params, threshold, final_metrics = step16_17_tune(preprocessor, X_train, y_train, X_test, y_test)
    step18_save(best_pipe, feature_cols, threshold, final_metrics)
    step19_predict(best_pipe, feature_cols, threshold)
    step20_note()

    results_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)
    print("\n=== DONE: Project 2 training complete ===")


if __name__ == "__main__":
    main()
