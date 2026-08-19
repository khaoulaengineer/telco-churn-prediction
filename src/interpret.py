import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

INPUT_PATH = "data/processed/customers_features.csv"
MODEL_PATH = "models/best_model.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"
OUTPUT_DIR = "docs"


def load_data_and_model():
    df = pd.read_csv(INPUT_PATH)
    y = df["churn"].astype(int)
    drop_cols = [c for c in ["churn", "customerid", "customerID"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = pd.get_dummies(X, drop_first=True).fillna(0)

    feature_columns = joblib.load(COLUMNS_PATH)
    X = X.reindex(columns=feature_columns, fill_value=0)

    # Forcer un typage numérique pur (float64) pour éviter les bugs SHAP
    # liés aux colonnes bool/category mélangées
    X = X.astype(np.float64)

    model = joblib.load(MODEL_PATH)
    return X, y, model


if __name__ == "__main__":
    X, y, model = load_data_and_model()

    _, X_sample, _, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Réduit l'échantillon pour accélérer le calcul (100 lignes suffisent pour l'interprétation)
    X_sample = X_sample.sample(n=min(200, len(X_sample)), random_state=42).reset_index(drop=True)

    print("Calcul des valeurs SHAP en cours...")

    # Choix explicite de l'explainer selon le type de modèle (plus stable que l'auto-détection)
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer(X_sample)
    else:
        # Random Forest / XGBoost -> TreeExplainer, plus rapide et plus fiable
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    print(f"Sauvegardé : {OUTPUT_DIR}/shap_summary.png")
    plt.close()

    # 2. Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_importance_bar.png", dpi=150, bbox_inches="tight")
    print(f"Sauvegardé : {OUTPUT_DIR}/shap_importance_bar.png")
    plt.close()


    idx = 0
    plt.figure()
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_example_client.png", dpi=150, bbox_inches="tight")
    print(f"Sauvegardé : {OUTPUT_DIR}/shap_example_client.png")
    plt.close()

    print("\nTerminé. Regarde shap_summary.png en premier.")