import pandas as pd
import numpy as np
import joblib

INPUT_PATH = "data/processed/customers_features.csv"
MODEL_PATH = "models/best_model.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"
OUTPUT_PATH = "data/processed/customers_scored.csv"

# Hypothèses business — à ajuster selon le contexte réel si tu veux affiner
COUT_ACTION_RETENTION = 50    # coût moyen d'une action de rétention (remise, appel, offre) par client
VALEUR_CLIENT_ANNUELLE = 700  # revenu annuel moyen généré par un client (à adapter selon monthlycharges * 12)


def load_data_and_model():
    df = pd.read_csv(INPUT_PATH)
    df_original = df.copy()

    y = df["churn"].astype(int)
    drop_cols = [c for c in ["churn", "customerid", "customerID"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = pd.get_dummies(X, drop_first=True).fillna(0)

    feature_columns = joblib.load(COLUMNS_PATH)
    X = X.reindex(columns=feature_columns, fill_value=0)
    X = X.astype(np.float64)

    model = joblib.load(MODEL_PATH)
    return df_original, X, y, model


def segment_risk(proba):
    if proba >= 0.6:
        return "Risque élevé"
    elif proba >= 0.3:
        return "Risque moyen"
    else:
        return "Risque faible"


if __name__ == "__main__":
    df, X, y, model = load_data_and_model()

    # Score de probabilité de churn pour chaque client
    df["proba_churn"] = model.predict_proba(X)[:, 1]
    df["segment_risque"] = df["proba_churn"].apply(segment_risk)

    print("="*60)
    print("SEGMENTATION DES CLIENTS PAR NIVEAU DE RISQUE")
    print("="*60)
    segment_counts = df["segment_risque"].value_counts()
    print(segment_counts)
    print(f"\nRépartition en % :")
    print((segment_counts / len(df) * 100).round(1))

    # Impact business estimé
    print("\n" + "="*60)
    print("IMPACT BUSINESS ESTIMÉ")
    print("="*60)

    nb_risque_eleve = (df["segment_risque"] == "Risque élevé").sum()
    nb_vrais_churners_dans_segment = df[
        (df["segment_risque"] == "Risque élevé") & (df["churn"] == 1)
    ].shape[0]

    print(f"Clients identifiés à risque élevé : {nb_risque_eleve}")
    print(f"Dont réellement churné (validation historique) : {nb_vrais_churners_dans_segment}")
    print(f"Précision du ciblage sur ce segment : {nb_vrais_churners_dans_segment/nb_risque_eleve*100:.1f}%")

    cout_total_campagne = nb_risque_eleve * COUT_ACTION_RETENTION
    valeur_clients_sauves_potentiel = nb_vrais_churners_dans_segment * VALEUR_CLIENT_ANNUELLE

    print(f"\nCoût d'une campagne de rétention ciblée sur ce segment : {cout_total_campagne:,.0f} DH")
    print(f"Valeur potentiellement sauvée (si 100% de rétention réussie) : {valeur_clients_sauves_potentiel:,.0f} DH")
    print(f"ROI potentiel (hypothèse haute) : {(valeur_clients_sauves_potentiel/cout_total_campagne - 1)*100:.0f}%")
    print("\n⚠️ Hypothèses de coûts/valeurs à ajuster selon données réelles de l'entreprise.")

    # Sauvegarde du dataset scoré
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset avec scores sauvegardé : {OUTPUT_PATH}")

    # Top variables par segment (pour recommandations)
    print("\n" + "="*60)
    print("PROFIL DU SEGMENT 'RISQUE ÉLEVÉ' (pour recommandations)")
    print("="*60)
    high_risk = df[df["segment_risque"] == "Risque élevé"]
    if "contract" in df.columns:
        print("\nType de contrat le plus fréquent :")
        print(high_risk["contract"].value_counts(normalize=True).round(2) * 100)
    if "tenure" in df.columns:
        print(f"\nAncienneté moyenne : {high_risk['tenure'].mean():.1f} mois")
    if "monthlycharges" in df.columns:
        print(f"Charges mensuelles moyennes : {high_risk['monthlycharges'].mean():.2f}")