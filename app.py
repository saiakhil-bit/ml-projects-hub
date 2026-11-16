"""
ML Projects Hub - one Streamlit app, two projects

  1. Lead Conversion Predictor   (Logistic Regression, classification)
  2. Campaign Revenue Predictor  (Linear/Ridge Regression, regression)

Run with:
    streamlit run app.py
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML Projects Hub", page_icon="🧠", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
LEAD_DIR = BASE_DIR / "projects" / "lead_conversion"
REV_DIR = BASE_DIR / "projects" / "revenue_prediction"

CHANNELS = ["Social", "Search", "Email", "Display", "Affiliate"]
PRODUCTS = ["Basic Plan", "Pro Plan", "Enterprise Plan", "Add-on"]

LEAD_RANGES = {
    "prior_website_visits": (0, 167),
    "prior_email_engagement_count": (0, 48),
}
REV_RANGES = {
    "planned_budget_inr": (1000, 190000),
    "planned_duration_days": (5, 60),
    "planned_audience_size": (2000, 500000),
    "offer_discount_pct": (0, 40),
}


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(model_file: str, meta_file: str):
    pipe = joblib.load(model_file)
    with open(meta_file) as f:
        meta = json.load(f)
    return pipe, meta


def range_warnings(values: dict, ranges: dict):
    out = []
    for field, value in values.items():
        lo, hi = ranges[field]
        if value < lo or value > hi:
            out.append(f"⚠️ {field} = {value} is outside the observed training range [{lo}, {hi}]")
    return out


def show_comparison(csv_path: Path):
    st.subheader("Model Comparison")
    try:
        st.dataframe(pd.read_csv(csv_path))
    except FileNotFoundError:
        st.info("Run train_model.py first to generate model_comparison.csv")


def download_button(df: pd.DataFrame, file_name: str, key: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download Predictions CSV",
        data=buf.getvalue(),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


def show_eda_images(folder: Path, key_prefix: str):
    imgs = sorted(folder.glob("*.png"))
    if not imgs:
        st.info("No EDA images found.")
        return
    cols = st.columns(2)
    for i, img in enumerate(imgs):
        with cols[i % 2]:
            st.image(str(img), caption=img.stem.replace("_", " ").title())


# ----------------------------------------------------------------------------
# Project 1 - Lead Conversion (Logistic Regression)
# ----------------------------------------------------------------------------
def lead_features(channel, product, visits, email):
    return pd.DataFrame([{
        "channel": channel,
        "product_interest": product,
        "prior_website_visits": visits,
        "prior_email_engagement_count": email,
        "total_engagement": visits + email,
        "engagement_ratio": (email / visits) if visits else 0,
    }])


def priority_label(prob, threshold):
    if prob >= max(threshold, 0.75):
        return "🔴 High priority"
    if prob >= threshold:
        return "🟡 Medium priority"
    return "⚪ Low priority"


def lead_page():
    pipe, meta = load_model(
        str(LEAD_DIR / "model" / "conversion_model_pipeline.pkl"),
        str(LEAD_DIR / "model" / "model_metadata.json"),
    )
    metrics = meta.get("metrics", {})
    threshold = meta.get("decision_threshold", 0.5)

    st.title("🎯 Lead Conversion Predictor")
    st.caption("Logistic Regression prototype · estimates the probability a lead converts within 30 days")

    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch CSV Prediction", "📊 EDA Plots"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lead Inputs")
            channel = st.selectbox("Channel", CHANNELS, key="lead_channel")
            product = st.selectbox("Product Interest", PRODUCTS, key="lead_product")
            visits = st.number_input("Prior Website Visits", min_value=0, value=20, step=1, key="lead_visits")
            email = st.number_input("Prior Email Engagement Count", min_value=0, value=15, step=1, key="lead_email")
            clicked = st.button("Predict Conversion", type="primary", key="lead_predict")

        with col2:
            st.subheader("Model Performance Summary")
            st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            m2.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            m3.metric("F1", f"{metrics.get('f1', 0):.4f}")
            m4, m5 = st.columns(2)
            m4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            m5.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
            st.caption(f"Decision threshold = {threshold:.2f} (selected on validation data)")

            if clicked:
                warns = range_warnings(
                    {"prior_website_visits": visits, "prior_email_engagement_count": email}, LEAD_RANGES
                )
                prob = pipe.predict_proba(lead_features(channel, product, visits, email))[0, 1]
                pred = int(prob >= threshold)
                st.success(f"### Conversion probability: {prob:.1%}  →  {priority_label(prob, threshold)}")
                st.write(f"Predicted class: **{'Will convert' if pred == 1 else 'Unlikely to convert'}**")
                for w in warns:
                    st.warning(w)

        st.divider()
        show_comparison(LEAD_DIR / "model" / "model_comparison.csv")

    with tab2:
        st.subheader("Batch Prediction via CSV Upload")
        st.caption("Upload a CSV with columns: channel, product_interest, prior_website_visits, prior_email_engagement_count")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="lead_upload")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            required = ["channel", "product_interest", "prior_website_visits", "prior_email_engagement_count"]
            missing = [c for c in required if c not in batch_df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
            else:
                rows = batch_df.dropna(subset=required).copy()
                dropped = len(batch_df) - len(rows)
                if dropped:
                    st.warning(f"⚠️ Dropped {dropped} rows with missing required values.")
                rows["total_engagement"] = rows["prior_website_visits"] + rows["prior_email_engagement_count"]
                rows["engagement_ratio"] = (
                    rows["prior_email_engagement_count"] / rows["prior_website_visits"].replace(0, np.nan)
                ).fillna(0)
                proba = pipe.predict_proba(rows[meta["feature_columns"]])[:, 1]
                rows["conversion_probability"] = proba.round(4)
                rows["follow_up_priority"] = [priority_label(p, threshold) for p in proba]
                st.dataframe(rows)
                download_button(rows, "lead_conversion_predictions.csv", "lead_download")

    with tab3:
        st.subheader("Exploratory Analysis & Model Plots")
        show_eda_images(LEAD_DIR / "assets", "lead")

    st.divider()
    with st.expander("ℹ️ About this model"):
        st.write(
            "This Logistic Regression model estimates the probability a lead converts within "
            "30 days, using engagement features measured before the scoring cutoff only. "
            "The decision threshold was chosen on validation data to balance precision and recall "
            "for marketing follow-up capacity."
        )


# ----------------------------------------------------------------------------
# Project 2 - Campaign Revenue (Linear Regression)
# ----------------------------------------------------------------------------
def revenue_features(budget, duration, audience, discount, channel):
    return pd.DataFrame([{
        "channel": channel,
        "planned_budget_inr": budget,
        "planned_duration_days": duration,
        "planned_audience_size": audience,
        "offer_discount_pct": discount,
        "budget_per_day": budget / duration if duration else 0,
        "audience_per_budget": audience / budget if budget else 0,
    }])


def revenue_page():
    pipe, meta = load_model(
        str(REV_DIR / "model" / "revenue_model_pipeline.pkl"),
        str(REV_DIR / "model" / "model_metadata.json"),
    )
    metrics = meta.get("metrics", {})

    st.title("📈 Campaign Revenue Predictor")
    st.caption("Linear Regression prototype · predicts estimated campaign revenue (INR) at the planning stage")

    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch CSV Prediction", "📊 EDA Plots"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Campaign Inputs")
            channel = st.selectbox("Channel", CHANNELS, key="rev_channel")
            budget = st.number_input("Planned Budget (INR)", min_value=0, value=150000, step=1000, key="rev_budget")
            duration = st.number_input("Planned Duration (days)", min_value=1, value=21, step=1, key="rev_duration")
            audience = st.number_input("Planned Audience Size", min_value=0, value=120000, step=1000, key="rev_audience")
            discount = st.slider("Offer Discount (%)", min_value=0.0, max_value=40.0, value=10.0, step=0.5, key="rev_discount")
            clicked = st.button("Predict Revenue", type="primary", key="rev_predict")

        with col2:
            st.subheader("Model Performance Summary")
            st.metric("Test R²", f"{metrics.get('R2', 0):.4f}")
            m1, m2 = st.columns(2)
            m1.metric("MAE (INR)", f"{metrics.get('MAE', 0):,.0f}")
            m2.metric("RMSE (INR)", f"{metrics.get('RMSE', 0):,.0f}")
            st.caption("Evaluated on a held-out test set not seen during training.")

            if clicked:
                warns = range_warnings(
                    {
                        "planned_budget_inr": budget,
                        "planned_duration_days": duration,
                        "planned_audience_size": audience,
                        "offer_discount_pct": discount,
                    },
                    REV_RANGES,
                )
                if duration <= 0:
                    st.error("❌ Invalid input: duration must be greater than 0.")
                else:
                    pred = pipe.predict(revenue_features(budget, duration, audience, discount, channel))[0]
                    st.success(f"### Estimated Revenue: ₹{pred:,.0f}")
                    for w in warns:
                        st.warning(w)

        st.divider()
        show_comparison(REV_DIR / "model" / "model_comparison.csv")

    with tab2:
        st.subheader("Batch Prediction via CSV Upload")
        st.caption(
            "Upload a CSV with columns: channel, planned_budget_inr, planned_duration_days, "
            "planned_audience_size, offer_discount_pct"
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="rev_upload")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            required = ["channel", "planned_budget_inr", "planned_duration_days",
                        "planned_audience_size", "offer_discount_pct"]
            missing = [c for c in required if c not in batch_df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
            else:
                rows = batch_df.dropna(subset=required).copy()
                dropped = len(batch_df) - len(rows)
                if dropped:
                    st.warning(f"⚠️ Dropped {dropped} rows with missing required values.")
                rows["budget_per_day"] = rows["planned_budget_inr"] / rows["planned_duration_days"].replace(0, np.nan)
                rows["audience_per_budget"] = rows["planned_audience_size"] / rows["planned_budget_inr"].replace(0, np.nan)
                rows = rows.dropna(subset=["budget_per_day", "audience_per_budget"])
                rows["predicted_revenue_inr"] = pipe.predict(rows[meta["feature_columns"]]).round(0)
                st.dataframe(rows)
                download_button(rows, "campaign_revenue_predictions.csv", "rev_download")

    with tab3:
        st.subheader("Exploratory Analysis Plots")
        show_eda_images(REV_DIR / "assets", "rev")

    st.divider()
    with st.expander("ℹ️ About this model"):
        st.write(
            "This Linear Regression model predicts estimated campaign revenue (INR) using "
            "planning-stage inputs only — no realized clicks, conversions or spend are used, "
            "to avoid leakage. Trained on cleaned, deduplicated campaign records with outliers "
            "removed via the IQR method."
        )


# ----------------------------------------------------------------------------
# Main - project selector
# ----------------------------------------------------------------------------
PROJECTS = {
    "🎯 Lead Conversion Predictor": lead_page,
    "📈 Campaign Revenue Predictor": revenue_page,
}


def main():
    st.sidebar.title("🧠 ML Projects Hub")
    st.sidebar.caption("Select a project")
    choice = st.sidebar.radio("Project", list(PROJECTS.keys()), key="project_choice", label_visibility="collapsed")
    st.sidebar.divider()
    if "Lead" in choice:
        st.sidebar.info("**Type:** Classification\n\n**Model:** Logistic Regression\n\n**Target:** converted_30d")
    else:
        st.sidebar.info("**Type:** Regression\n\n**Model:** Ridge / Linear Regression\n\n**Target:** revenue_inr")
    PROJECTS[choice]()


if __name__ == "__main__":
    main()
