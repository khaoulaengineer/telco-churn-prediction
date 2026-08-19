import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score,
    recall_score, f1_score
)
from xgboost import XGBClassifier
import joblib
import os

INPUT_PATH = "data/processed/customers_features.csv"
MODEL_OUTPUT_DIR = "models"


def load_and_prepare(path: str):
    df = pd.read_csv(path)

    # Cible
    y = df["churn"].astype(int) if df["churn"].dtype != bool else df["churn"].astype(int)

    # On retire les colonnes non prédictives ou déjà utilisées ailleurs
    drop_cols = [c for c in ["churn", "customerid", "customerID"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Encodage des variables catégorielles
    X = pd.get_dummies(X, drop_first=True)

    # Gestion des éventuels NaN restants (ex: tenure_segment si pandas.cut a créé des NaN)
    X = X.fillna(0)

    return X, y


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"Modèle : {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Non-churn", "Churn"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

    return {
        "model": name,
        "precision_churn": precision_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
        "f1_churn": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


if __name__ == "__main__":
    X, y = load_and_prepare(INPUT_PATH)
    print(f"Features : {X.shape[1]} colonnes | Observations : {X.shape[0]}")
    print(f"Taux de churn : {y.mean()*100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []

    # 1. Baseline — Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_reg.fit(X_train, y_train)
    results.append(evaluate_model("Logistic Regression", log_reg, X_test, y_test))

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    # 3. XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate_model("XGBoost", xgb, X_test, y_test))

    # Comparaison finale
    print(f"\n{'='*50}")
    print("COMPARAISON DES MODÈLES")
    print(f"{'='*50}")
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print(results_df.to_string(index=False))

    # Sauvegarde du meilleur modèle
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    best_model_name = results_df.iloc[0]["model"]
    best_model = {"Logistic Regression": log_reg, "Random Forest": rf, "XGBoost": xgb}[best_model_name]
    joblib.dump(best_model, f"{MODEL_OUTPUT_DIR}/best_model.pkl")
    print(f"\nMeilleur modèle ({best_model_name}) sauvegardé dans {MODEL_OUTPUT_DIR}/best_model.pkl")

    # Sauvegarde des colonnes utilisées (nécessaire pour réutiliser le modèle plus tard)
    joblib.dump(list(X.columns), f"{MODEL_OUTPUT_DIR}/feature_columns.pkl")