import os 
import urllib.request 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier 
import joblib

DATA_DIR = "data"
MODELS_DIR = "models"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = os.path.join(DATA_DIR, "telco_churn.csv")

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Telco Churn dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Download Complete.")

def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    # Clean whitespace in TotalCharges and convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    df = df.dropna(subset=['TotalCharges'])

    # Map target: Yes -> 1, No -> 0 
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop identifier column 
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    return df 

def build_and_train():
    download_data()
    df = load_and_clean_data()

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Separate numerical and categorical columns 
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print("Training XGBoost Pipeline....")
    pipeline.fit(X_train, y_train)

    # Evaluate 
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*40)
    print("Model Performance Evaluation")
    print("="*40)
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("="*40)

    # Export trained pipeline 
    model_save_path = os.path.join(MODELS_DIR, "churn_pipeline.joblib")
    joblib.dump(pipeline, model_save_path)
    print(f"\nModel Pipeline saved to: {model_save_path}\n")

if __name__ == "__main__":
    build_and_train()