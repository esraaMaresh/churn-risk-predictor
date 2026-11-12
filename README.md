# 🎯 Customer Retention & Churn Intelligence System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and decision-support system that predicts customer churn risk, identifies key drivers of customer attrition, and provides real-time retention recommendations.

---

## 📌 Problem & Business Context

Customer acquisition costs **5x to 7x more** than retaining existing customers. In subscription businesses, losing customers directly diminishes customer lifetime value (LTV) and compound revenue growth.

This project addresses two critical challenges:
1. **Early Risk Identification**: Accurately spotting churn risk *before* the customer cancels.
2. **Actionable Intervention**: Surfacing actionable risk levels so retention teams can proactively intervene with customized offers.

---

## 🏗️ System Architecture & Workflow

```text
churn-risk-predictor/
├── data/
│   └── telco_churn.csv              # Raw IBM Telco dataset
├── models/
│   └── churn_pipeline.joblib        # Exported Scikit-Learn + XGBoost pipeline
├── src/
│   └── pipeline.py                  # Ingestion, preprocessing & model training
├── app/
│   └── streamlit_app.py             # Interactive decision-support web UI
├── .gitignore                       # Git exclusion rules
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

### End-to-End Workflow:


---

## 📊 Model Performance & Evaluation

The dataset exhibits typical class imbalance (~26.5% churn rate). The model uses `scale_pos_weight` to prioritize catching positive churn cases without sacrificing general accuracy.

| Metric | Score | Business Implication |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8392** | Strong discriminative ability across decision thresholds |
| **Recall (Churn / Class 1)** | **0.80** | Successfully flags **4 out of every 5** churn-bound accounts |
| **Precision (Non-Churn)** | **0.91** | High confidence when predicting customer stability |
| **Overall Accuracy** | **0.73** | Well-balanced across both majority and minority classes |

> **Why Recall Matters Most Here**: In customer churn, a **False Negative** (failing to detect an account that leaves) causes irreversible revenue loss, whereas a **False Positive** merely results in an extra proactive check-in or promotional email.

---

## 🛠️ Tech Stack & Engineering Practices

- **Core ML**: `Python`, `Scikit-Learn`, `XGBoost`, `Pandas`, `NumPy`
- **Data Preprocessing**: `ColumnTransformer` with `StandardScaler` for continuous variables and `OneHotEncoder` for categorical features.
- **Application & UI**: `Streamlit`
- **Model Persistence**: `Joblib`
- **Engineering Principles**: Modular structure separating training logic (`src/`) from application delivery (`app/`), fully reproducible pipeline.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ installed

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/churn-risk-predictor.git
cd churn-risk-predictor

### 3 Setting up the envrionment
# Windows
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

. Install Dependencies
bash
pip install -r requirements.txt
5. Train the Model Pipeline
bash
python src/pipeline.py

(Automatically fetches the data, trains the pipeline, prints evaluation metrics, and saves models/churn_pipeline.joblib).

6. Run the Interactive Dashboard
bash
streamlit run app/streamlit_app.py

Your browser will open automatically at http://localhost:8501.