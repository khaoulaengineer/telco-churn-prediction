import pandas as pd

INPUT_PATH = "data/processed/customers_clean.csv"
OUTPUT_PATH = "data/processed/customers_features.csv"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratio charges mensuelles / charges totales (attention division par zéro pour tenure=0)
    if "monthlycharges" in df.columns and "totalcharges" in df.columns:
        df["charges_ratio"] = df["monthlycharges"] / df["totalcharges"].replace(0, 1)

    # Nombre de services souscrits (basé sur tes colonnes réelles)
    service_cols = [
        c for c in [
            "onlinesecurity", "onlinebackup", "deviceprotection",
            "techsupport", "streamingtv", "streamingmovies"
        ] if c in df.columns
    ]
    if service_cols:
        df["nb_services"] = df[service_cols].apply(
            lambda row: sum(1 for v in row if v == True or v == "Yes"), axis=1
        )

    # Segmentation d'ancienneté — confirmé comme variable clé par ton EDA
    if "tenure" in df.columns:
        df["tenure_segment"] = pd.cut(
            df["tenure"],
            bins=[-1, 6, 12, 24, 48, 100],
            labels=["0-6m", "6-12m", "1-2ans", "2-4ans", "4ans+"]
        )

    # Flag "client à risque contractuel" — basé sur ton observation Month-to-month = 43% churn
    if "contract" in df.columns:
        df["contrat_mensuel"] = (df["contract"] == "Month-to-month").astype(int)

    # Flag "client récent" — basé sur ton observation tenure faible = churn élevé
    if "tenure" in df.columns:
        df["client_recent"] = (df["tenure"] <= 6).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"Données chargées : {df.shape}")

    df = add_engineered_features(df)
    print(f"Features ajoutées. Shape après feature engineering : {df.shape}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Sauvegardé : {OUTPUT_PATH}")

    print("\nAperçu des nouvelles colonnes :")
    new_cols = ["charges_ratio", "nb_services", "tenure_segment", "contrat_mensuel", "client_recent"]
    print(df[[c for c in new_cols if c in df.columns]].head())