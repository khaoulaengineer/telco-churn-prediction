# 📊 Churn Prediction — Telco Customer Churn

Prédiction du churn client pour un opérateur télécom, avec une approche complète : ingénierie des données, machine learning interprétable (SHAP), et traduction en impact business chiffré.

## Contexte business

Un client qui résilie coûte cher : il faut en acquérir un nouveau pour le remplacer, à un coût généralement bien supérieur à celui de la rétention. Ce projet identifie **à l'avance** les clients les plus susceptibles de partir, pour permettre une action de rétention ciblée plutôt qu'une approche généraliste coûteuse.

## Résultats clés

| Indicateur | Valeur |
|---|---|
| Taux de churn global | **26.5 %** |
| Clients identifiés "risque élevé" | 2 326 |
| Précision du ciblage sur ce segment | **57.5 %** |
| ROI potentiel estimé (campagne de rétention ciblée) | **+705 %** |
| Modèle retenu | Logistic Regression (meilleur ROC-AUC) |

**Principal facteur de churn identifié :** le type de contrat. Les clients en engagement **mensuel (Month-to-month)** churnent massivement plus que ceux en contrat 1 ou 2 ans — confirmé à la fois par l'analyse exploratoire et par les valeurs SHAP. L'ancienneté (tenure) est le deuxième facteur clé : le risque est concentré sur les 6 premiers mois de la relation client.

## Démarche

1. **Ingénierie des données** — nettoyage, typage, gestion des valeurs manquantes (`src/ingest.py`)
2. **Exploration (EDA)** — distribution du churn, corrélations, visualisations (`notebooks/eda_churn.ipynb`)
3. **Feature engineering** — segmentation d'ancienneté, ratio de charges, nombre de services souscrits, flags métier (`src/features.py`)
4. **Modélisation** — comparaison de 3 modèles (Logistic Regression, Random Forest, XGBoost) avec gestion du déséquilibre de classes et validation stratifiée (`src/train_model.py`)
5. **Interprétabilité** — analyse SHAP pour expliquer les prédictions, au niveau global et individuel (`src/interpret.py`)
6. **Valorisation business** — scoring de risque par client, segmentation, estimation du ROI d'une campagne de rétention (`src/business_value.py`)

## Stack technique

Python · pandas · scikit-learn · XGBoost · SHAP · PostgreSQL · Jupyter

## Structure du projet

```
churn-prediction/
├── data/
│   ├── raw/                    # Dataset brut (Telco Customer Churn, IBM/Kaggle)
│   └── processed/              # Données nettoyées, avec features, scorées
├── sql/
│   └── schema.sql              # Schéma PostgreSQL
├── src/
│   ├── ingest.py                # Nettoyage et chargement des données
│   ├── features.py                # Feature engineering
│   ├── train_model.py               # Entraînement et comparaison des modèles
│   ├── interpret.py                   # Interprétabilité SHAP
│   └── business_value.py                # Scoring et impact business
├── notebooks/
│   └── eda_churn.ipynb                    # Analyse exploratoire
├── docs/
│   ├── shap_summary.png                     # Visualisations SHAP
│   ├── shap_importance_bar.png
│   └── shap_example_client.png
├── models/
│   └── best_model.pkl                         # Modèle final sauvegardé
└── requirements.txt
```

## Reproduire le projet

```bash
# Environnement
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Pipeline complet
python src/ingest.py
python src/features.py
python src/train_model.py
python src/interpret.py
python src/business_value.py
```

## Pistes d'amélioration

- Orchestration du pipeline avec Airflow
- API FastAPI pour scorer de nouveaux clients en temps réel
- Dashboard de suivi (Streamlit / Power BI)
- Test d'hypothèses de coûts business avec des données réelles d'entreprise

---

Projet réalisé par **Khaoula** — étudiante ingénieure en Data Engineering à l'ENSAH, Université Abdelmalek Essaâdi.