-- Schéma de la base pour le projet Churn Prediction
-- Adapté au dataset Telco Customer Churn

CREATE TABLE IF NOT EXISTS customers (
    customer_id         VARCHAR(20) PRIMARY KEY,
    gender               VARCHAR(10),
    senior_citizen       BOOLEAN,
    partner               BOOLEAN,
    dependents            BOOLEAN,
    tenure_months         INTEGER,
    phone_service         BOOLEAN,
    multiple_lines        VARCHAR(30),
    internet_service       VARCHAR(30),
    online_security         VARCHAR(30),
    online_backup           VARCHAR(30),
    device_protection       VARCHAR(30),
    tech_support             VARCHAR(30),
    streaming_tv               VARCHAR(30),
    streaming_movies             VARCHAR(30),
    contract_type                  VARCHAR(30),
    paperless_billing                BOOLEAN,
    payment_method                     VARCHAR(50),
    monthly_charges                      NUMERIC(10,2),
    total_charges                          NUMERIC(10,2),
    churn                                    BOOLEAN,
    inserted_at                                TIMESTAMP DEFAULT NOW()
);

-- Index utile pour les analyses par segment
CREATE INDEX IF NOT EXISTS idx_customers_contract ON customers(contract_type);
CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers(churn);

-- Exemple de requête d'analyse : taux de churn par type de contrat
-- SELECT contract_type, COUNT(*) AS total, SUM(CASE WHEN churn THEN 1 ELSE 0 END) AS churners,
--        ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
-- FROM customers
-- GROUP BY contract_type
-- ORDER BY churn_rate_pct DESC;
