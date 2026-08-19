# Roadmap détaillée — Churn Prediction

## Phase 0 — Cadrage (1-2 jours)
- [ ] Définir la problématique business précise
- [ ] Choisir le dataset final (Telco recommandé pour démarrer)
- [ ] Définir les métriques : recall classe churn, precision, ROC-AUC (pas accuracy seule, classes déséquilibrées)
- [ ] Estimer le coût métier : coût d'un faux négatif (client perdu non détecté) vs faux positif (action de rétention "gaspillée")

## Phase 1 — Ingénierie des données (3-5 jours)
- [ ] Créer une base PostgreSQL locale (ou Supabase/Neon gratuit si tu veux du cloud)
- [ ] Écrire `sql/schema.sql` pour la table clients
- [ ] Script `src/ingest.py` : lecture CSV → nettoyage basique → insertion en base
- [ ] Nettoyage : types de colonnes (TotalCharges souvent en string dans Telco !), valeurs manquantes, doublons
- [ ] Documenter chaque décision de nettoyage dans un notebook ou log

## Phase 2 — EDA & Feature Engineering (4-6 jours)
- [ ] Notebook EDA : distribution du churn (%), corrélations avec les variables
- [ ] Visualisations : churn par type de contrat, ancienneté, méthode de paiement, services souscrits
- [ ] Feature engineering :
  - Ancienneté en mois (tenure)
  - Ratio charges mensuelles / charges totales
  - Nombre de services souscrits (agrégation)
  - Encodage des variables catégorielles (one-hot ou target encoding)
- [ ] Gestion du déséquilibre : SMOTE, ou `class_weight='balanced'` dans les modèles

## Phase 3 — Modélisation (5-7 jours)
- [ ] Split train/test stratifié (important vu le déséquilibre)
- [ ] Baseline : Logistic Regression
- [ ] Random Forest
- [ ] XGBoost ou LightGBM
- [ ] Cross-validation (StratifiedKFold)
- [ ] Tuning : GridSearchCV ou Optuna
- [ ] Comparaison des modèles : tableau récapitulatif (Precision, Recall, F1, ROC-AUC)
- [ ] Interprétabilité : SHAP values (summary plot + force plot sur cas individuels)

## Phase 4 — Valorisation business (2-3 jours)
- [ ] Traduire les résultats en impact business chiffré (ex : "X% des churners identifiés → économie estimée de Y DH")
- [ ] Segmenter les clients à risque (score de risque faible/moyen/élevé)
- [ ] Rédiger 3-5 recommandations d'action concrètes par segment

## Phase 5 — Industrialisation (optionnel, 3-5 jours)
- [ ] API FastAPI : endpoint `/predict` qui prend un profil client et retourne une probabilité de churn
- [ ] Dashboard Streamlit ou Power BI pour visualiser les prédictions
- [ ] (Bonus, aligné avec ta roadmap 18 mois) Orchestration simple avec Airflow

## Phase 6 — Packaging & storytelling (2 jours)
- [ ] README final soigné (comme celui déjà généré)
- [ ] Nettoyer le repo GitHub, vérifier qu'il n'y a pas de données sensibles commitées
- [ ] Rapport de synthèse (1-2 pages, PDF) — contexte, méthodo, résultats, recommandations
- [ ] Post LinkedIn de valorisation du projet
