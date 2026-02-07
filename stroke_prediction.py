# ================================
# Stroke Prediction ML Project
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier


def main():
    # -------------------------------
    # Load Dataset
    # -------------------------------
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    print("Dataset Loaded Successfully!")
    print(df.head())

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    # Drop ID column
    df.drop("id", axis=1, inplace=True)

    # Handle missing BMI
    df["bmi"].fillna(df["bmi"].median(), inplace=True)

    # Encode categorical columns
    categorical_cols = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # -------------------------------
    # Split Features & Target
    # -------------------------------
    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -------------------------------
    # Handle Class Imbalance
    # -------------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # -------------------------------
    # Train XGBoost Model
    # -------------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # Evaluation
    # -------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n================ RESULTS ================")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
