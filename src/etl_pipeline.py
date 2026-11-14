import os
import json
import logging
import sqlite3
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BreastCancerETLPipeline")


def generate_source_feed(raw_path="data/raw/breast_cancer_raw_ingest.csv"):
    """Simulate upstream clinical biopsy data feed if not already present."""
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    data = load_breast_cancer()
    raw_df = pd.DataFrame(data.data, columns=data.feature_names)
    raw_df["patient_id"] = [f"PT-{1000 + i}" for i in range(len(raw_df))]
    raw_df["tumor_size_grade"] = pd.qcut(
        raw_df["mean radius"], q=3, labels=["Small", "Medium", "Large"]
    ).astype(str)
    raw_df["texture_pattern"] = np.where(
        raw_df["mean texture"] < 17.0, "Smooth",
        np.where(raw_df["mean texture"] < 21.5, "Irregular", "Rough")
    )
    raw_df["diagnosis_label"] = np.where(data.target == 0, "Malignant", "Benign")
    raw_df["target"] = data.target

    rng = np.random.RandomState(42)
    raw_df.loc[rng.choice(raw_df.index, size=15, replace=False), "mean smoothness"] = np.nan
    raw_df.loc[rng.choice(raw_df.index, size=12, replace=False), "mean compactness"] = np.nan

    dup_rows = raw_df.iloc[:15].copy()
    raw_source_df = pd.concat([raw_df, dup_rows], ignore_index=True)
    raw_source_df.to_csv(raw_path, index=False)
    logger.info(f"Source feed generated at {raw_path} ({raw_source_df.shape[0]} rows, {raw_source_df.shape[1]} cols)")
    return raw_path


def extract_data(raw_path="data/raw/breast_cancer_raw_ingest.csv"):
    """Stage 1: Extract raw CSV data from upstream source."""
    if not os.path.exists(raw_path):
        generate_source_feed(raw_path)
    df = pd.read_csv(raw_path)
    logger.info(f"[EXTRACT] Loaded {len(df)} rows and {df.shape[1]} columns from {raw_path}.")
    return df


def clean_and_validate(df, staging_path="data/staging/breast_cancer_cleaned_staging.csv"):
    """Stage 2: Deduplicate records, impute missing values, cap IQR outliers, and run quality checks."""
    os.makedirs(os.path.dirname(staging_path), exist_ok=True)
    initial_rows = len(df)
    dup_count = int(df.duplicated(subset=["patient_id"]).sum())
    missing_count = int(df.isnull().sum().sum())

    # 1. Deduplicate by patient_id
    cleaned_df = df.drop_duplicates(subset=["patient_id"], keep="first").copy()

    # 2. Median imputation for numeric columns with missing values
    for col in ["mean smoothness", "mean compactness"]:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

    # 3. IQR Outlier Capping across 30 core cytological features
    numeric_cols = [
        c for c in cleaned_df.select_dtypes(include=[np.number]).columns
        if c not in ["target"]
    ]
    outliers_capped = 0
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers_capped += int(((cleaned_df[col] < lower) | (cleaned_df[col] > upper)).sum())
        cleaned_df[col] = np.clip(cleaned_df[col], lower, upper)

    # Data Quality Gate Assertions
    assert cleaned_df.isnull().sum().sum() == 0, "Data Quality Gate Failed: Null values remain!"
    assert cleaned_df["patient_id"].is_unique, "Data Quality Gate Failed: Duplicate patient_ids remain!"

    cleaned_df.to_csv(staging_path, index=False)
    logger.info(
        f"[CLEAN & VALIDATE] Deduplicated {initial_rows} -> {len(cleaned_df)} rows (-{dup_count} dups), "
        f"imputed {missing_count} missing cells, capped {outliers_capped} outlier values."
    )
    return cleaned_df, dup_count, missing_count, outliers_capped


def transform_and_engineer(cleaned_df, processed_csv="data/processed/breast_cancer_warehouse_ready.csv"):
    """Stage 3: Domain feature engineering, categorical encoding, and Z-score scaling."""
    os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
    trans_df = cleaned_df.copy()

    # 1. Domain Feature Engineering
    trans_df["area_perimeter_ratio"] = trans_df["mean area"] / (trans_df["mean perimeter"] + 1e-6)
    trans_df["radius_worst_to_mean_ratio"] = trans_df["worst radius"] / (trans_df["mean radius"] + 1e-6)
    trans_df["concavity_smoothness_index"] = trans_df["mean concavity"] * trans_df["mean smoothness"]
    trans_df["circularity_deviation"] = np.abs(
        (trans_df["mean perimeter"] / (2 * np.pi * trans_df["mean radius"] + 1e-6)) - 1.0
    )
    trans_df["nuclear_malformation_score"] = (
        trans_df["worst concavity"] + trans_df["worst concave points"] + trans_df["worst compactness"]
    )

    # 2. Categorical Encoding (Ordinal + One-Hot)
    trans_df["tumor_size_grade_encoded"] = trans_df["tumor_size_grade"].map({"Small": 0, "Medium": 1, "Large": 2})
    ohe = OneHotEncoder(sparse_output=False)
    ohe_arr = ohe.fit_transform(trans_df[["texture_pattern"]])
    ohe_cols = ohe.get_feature_names_out(["texture_pattern"])
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=trans_df.index)

    # 3. Standardize continuous features
    exclude = {"patient_id", "tumor_size_grade", "texture_pattern", "diagnosis_label", "target", "tumor_size_grade_encoded"}
    scale_cols = [c for c in trans_df.columns if c not in exclude]
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(trans_df[scale_cols]),
        columns=[f"{c}_scaled" for c in scale_cols],
        index=trans_df.index,
    )

    final_df = pd.concat([
        trans_df[["patient_id", "diagnosis_label", "target", "tumor_size_grade", "tumor_size_grade_encoded", "texture_pattern"]],
        ohe_df,
        trans_df[scale_cols],
        scaled_df,
    ], axis=1)

    final_df.columns = [c.replace(" ", "_") for c in final_df.columns]
    final_df.to_csv(processed_csv, index=False)
    logger.info(f"[TRANSFORM] Engineered & scaled dataset saved to {processed_csv} ({final_df.shape[0]} rows, {final_df.shape[1]} cols).")
    return final_df, len(scale_cols)


def load_to_warehouse(final_df, db_path="data/processed/clinical_data_warehouse.db"):
    """Stage 4: Load processed dataset into SQLite Data Warehouse & build analytical summary table."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    final_df.to_sql("fact_breast_cancer_diagnostics", conn, if_exists="replace", index=False)

    summary_sql = """
    SELECT
        diagnosis_label,
        tumor_size_grade,
        COUNT(*) AS patient_count,
        ROUND(AVG(mean_radius), 3) AS avg_mean_radius,
        ROUND(AVG(worst_radius), 3) AS avg_worst_radius,
        ROUND(AVG(area_perimeter_ratio), 3) AS avg_area_perimeter_ratio,
        ROUND(AVG(nuclear_malformation_score), 3) AS avg_malformation_score
    FROM fact_breast_cancer_diagnostics
    GROUP BY diagnosis_label, tumor_size_grade
    ORDER BY diagnosis_label, tumor_size_grade;
    """
    summary_df = pd.read_sql_query(summary_sql, conn)
    summary_df.to_sql("agg_diagnosis_by_size_grade", conn, if_exists="replace", index=False)
    conn.close()

    os.makedirs("reports", exist_ok=True)
    summary_df.to_csv("reports/warehouse_summary_query.csv", index=False)
    logger.info(f"[LOAD] Loaded {len(final_df)} rows into SQLite warehouse ({db_path}) and built aggregation table.")
    return summary_df


def run_pipeline():
    """Execute end-to-end ETL pipeline and write audit log."""
    raw_df = extract_data()
    cleaned_df, dup_cnt, miss_cnt, outlier_cnt = clean_and_validate(raw_df)
    final_df, scaled_cnt = transform_and_engineer(cleaned_df)
    summary_df = load_to_warehouse(final_df)

    audit = {
        "pipeline_name": "breast_cancer_clinical_etl_pipeline",
        "executed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "SUCCESS",
        "rows_extracted": int(len(raw_df)),
        "duplicates_removed": dup_cnt,
        "missing_values_imputed": miss_cnt,
        "outliers_iqr_capped": outlier_cnt,
        "final_rows_loaded": int(len(final_df)),
        "final_columns_loaded": int(final_df.shape[1]),
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/pipeline_execution_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print("\n=== SQLITE DATA WAREHOUSE ANALYTICAL SUMMARY ===")
    print(summary_df.to_string(index=False))
    return audit


if __name__ == "__main__":
    run_pipeline()
