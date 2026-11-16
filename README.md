# ML Projects Hub

One Streamlit app (`app.py`) that combines two ML projects. Pick a project in the
sidebar and its own input form, prediction, batch-CSV tab and EDA plots open.

| Project | Model | Task |
|---|---|---|
| Lead Conversion Predictor | Logistic Regression | Classification (converted in 30 days) |
| Campaign Revenue Predictor | Ridge / Linear Regression | Regression (revenue in INR) |

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure
```
ml_projects_hub/
├── app.py                      # single Streamlit app (project selector)
├── requirements.txt
└── projects/
    ├── lead_conversion/        # data, model, assets, notebooks, train_model.py
    └── revenue_prediction/     # data, model, assets, notebooks, train_model.py
```
Each project's original README, data dictionary, training script and notebook are kept inside its folder.
