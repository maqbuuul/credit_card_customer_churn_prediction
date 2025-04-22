-- 1. CREATE AND LOAD RAW
CREATE OR REPLACE TABLE customer_raw AS
SELECT * 
FROM read_csv_auto('D:\portfolio_projects\customer-churn-analysis\data\raw\Telecom_Data.csv', AUTO_DETECT TRUE, SAMPLE_SIZE=-1);

-- 2. CONVERT TYPES & CREATE STAGING
CREATE OR REPLACE TABLE customer_stg AS
SELECT
  CustomerID                           AS customer_id,
  CAST(Month AS DATE)                  AS month,
  CAST("Month of Joining" AS DATE)     AS join_month,
  CAST(Zip_code AS INTEGER)            AS zip_code,
  Gender                               AS gender,
  CAST(Age AS INTEGER)                 AS age,
  Married                              AS is_married,
  CAST(Dependents AS INTEGER)          AS num_dependents,
  country, state, county, timezone,
  CAST(latitude AS DOUBLE)             AS latitude,
  CAST(longitude AS DOUBLE)            AS longitude,
  CAST(arpu AS DOUBLE)                 AS arpu,
  CAST(roam_ic AS DOUBLE)              AS roam_ic,
  CAST(roam_og AS DOUBLE)              AS roam_og,
  CAST(vol_4g AS DOUBLE)               AS vol_4g,
  CAST(vol_5g AS DOUBLE)               AS vol_5g,
  CAST(total_rech_amt AS DOUBLE)       AS total_rech_amt,
  CAST(total_rech_data AS DOUBLE)      AS total_rech_data,
  aug_vbc_5g,
  "Streaming TV"           AS streaming_tv,
  "Streaming Movies"       AS streaming_movies,
  "Streaming Music"        AS streaming_music,
  "Internet Service"       AS internet_service,
  "Internet Type"          AS internet_type,
  Payment_Method           AS payment_method,
  CAST("Satisfaction Score" AS DOUBLE) AS satisfaction_score,
  CAST("Churn Value" AS INTEGER)       AS churn_flag
FROM customer_raw;

-- 3. HANDLE MISSING VALUES
UPDATE customer_stg 
SET
  age = COALESCE(age, 0),
  num_dependents = COALESCE(num_dependents, 0),
  arpu = COALESCE(arpu, 0),
  vol_4g = COALESCE(vol_4g, 0),
  vol_5g = COALESCE(vol_5g, 0),
  total_rech_amt = COALESCE(total_rech_amt, 0),
  total_rech_data = COALESCE(total_rech_data, 0);

UPDATE customer_stg
SET
  gender = COALESCE(gender, 'Unknown'),
  is_married = COALESCE(is_married, 'No'),
  internet_service = COALESCE(internet_service, 'None'),
  internet_type = COALESCE(internet_type, 'None'),
  payment_method = COALESCE(payment_method, 'Unknown');

-- 4. OUTLIER TREATMENT (IQR method)
WITH stats AS (
  SELECT
    PERCENTILE_CONT(arpu, 0.25) OVER () AS q1_arpu,
    PERCENTILE_CONT(arpu, 0.75) OVER () AS q3_arpu,
    PERCENTILE_CONT(total_rech_amt, 0.25) OVER () AS q1_rech,
    PERCENTILE_CONT(total_rech_amt, 0.75) OVER () AS q3_rech
  FROM customer_stg LIMIT 1
)
CREATE OR REPLACE TABLE customer_clean AS
SELECT
  cs.*,
  LEAST(GREATEST(cs.arpu, s.q1_arpu - 1.5 * (s.q3_arpu - s.q1_arpu)),
        s.q3_arpu + 1.5 * (s.q3_arpu - s.q1_arpu)) AS arpu_capped,
  LEAST(GREATEST(cs.total_rech_amt, s.q1_rech - 1.5 * (s.q3_rech - s.q1_rech)),
        s.q3_rech + 1.5 * (s.q3_rech - s.q1_rech)) AS rech_amt_capped
FROM customer_stg cs, stats s;

-- 5. FEATURE ENGINEERING
CREATE OR REPLACE TABLE customer_feat AS
SELECT
  customer_id,
  month,
  join_date,
  DATE_DIFF('month', join_date, month) AS tenure_months,
  churn_flag,
  arpu_capped AS arpu,
  rech_amt_capped AS total_rech_amt,
  COALESCE(roam_ic, 0) + COALESCE(roam_og, 0) AS total_out_min,
  CASE 
    WHEN (vol_4g + vol_5g) = 0 THEN 0 
    ELSE vol_5g / (vol_4g + vol_5g) 
  END AS data_5g_share,
  CAST(night_pck_user AS INTEGER)
  + CAST(fb_user AS INTEGER)
  + CAST(aug_vbc_5g AS INTEGER)
  + CAST(streaming_tv AS INTEGER)
  + CAST(streaming_movies AS INTEGER)
  + CAST(streaming_music AS INTEGER)
  + CAST(CASE WHEN internet_service <> 'None' THEN 1 ELSE 0 END AS INTEGER) AS add_on_count
FROM customer_clean;

-- 6. AGGREGATED CHURN RATE
CREATE OR REPLACE TABLE churn_by_tenure AS
SELECT
  CASE 
    WHEN tenure_months <= 3 THEN '0-3'
    WHEN tenure_months <=12 THEN '4-12'
    WHEN tenure_months <=24 THEN '13-24'
    ELSE '25+'
  END AS tenure_bucket,
  COUNT(*) AS customer_count,
  ROUND(SUM(churn_flag) * 100.0 / COUNT(*), 2) AS churn_rate
FROM customer_feat
GROUP BY 1
ORDER BY 1;
