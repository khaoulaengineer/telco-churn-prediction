import pandas as pd
from sqlalchemy import create_engine
import os

RAW_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_PATH = "data/processed/customers_clean.csv"

# Adapte cette URL à ta config PostgreSQL locale (ou laisse commenté si tu veux juste bosser en CSV pour l'instant)
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/churn_db")


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Renommer les colonnes en snake_case pour cohérence avec le schéma SQL
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # TotalCharges est souvent lu comme string dans le dataset Telco (espaces vides pour les nouveaux clients)
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
        df["totalcharges"] = df["totalcharges"].fillna(0)

    # Conversion des colonnes Yes/No en booléens
    yes_no_cols = [c for c in df.columns if df[c].dropna().isin(["Yes", "No"]).all()]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": True, "No": False})

    # Suppression des doublons éventuels
    before = len(df)
    df = df.drop_duplicates()
    print(f"Doublons supprimés : {before - len(df)}")

    return df


def save_processed(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Données nettoyées sauvegardées : {path}")


def load_to_postgres(df: pd.DataFrame, table_name: str = "customers"):
    try:
        engine = create_engine(DB_URL)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Données insérées dans la table '{table_name}'")
    except Exception as e:
        print(f"Connexion PostgreSQL non disponible, étape ignorée : {e}")


if __name__ == "__main__":
    df_raw = load_raw_data(RAW_PATH)
    df_clean = clean_data(df_raw)
    save_processed(df_clean, PROCESSED_PATH)
    load_to_postgres(df_clean)
