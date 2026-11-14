"""
Apache Airflow DAG for Orchestrating the Breast Cancer Clinical Data Pipeline.
Place in Airflow's dags/ directory or run standalone via `python src/etl_pipeline.py`.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.etl_pipeline import extract_data, clean_and_validate, transform_and_engineer, load_to_warehouse

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner": "data_science_intern",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }

    def task_extract():
        extract_data("data/raw/breast_cancer_raw_ingest.csv")

    def task_clean():
        df = extract_data("data/raw/breast_cancer_raw_ingest.csv")
        clean_and_validate(df, "data/staging/breast_cancer_cleaned_staging.csv")

    def task_transform():
        import pandas as pd
        staging_df = pd.read_csv("data/staging/breast_cancer_cleaned_staging.csv")
        transform_and_engineer(staging_df, "data/processed/breast_cancer_warehouse_ready.csv")

    def task_load():
        import pandas as pd
        processed_df = pd.read_csv("data/processed/breast_cancer_warehouse_ready.csv")
        load_to_warehouse(processed_df, "data/processed/clinical_data_warehouse.db")

    with DAG(
        dag_id="breast_cancer_clinical_etl_pipeline",
        default_args=default_args,
        description="End-to-end Extract, Clean, Transform, and SQLite Warehouse Load pipeline",
        schedule_interval="@daily",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["data_science", "etl", "breast_cancer"],
    ) as dag:

        t1_extract = PythonOperator(task_id="extract_clinical_source", python_callable=task_extract)
        t2_clean = PythonOperator(task_id="clean_and_validate_staging", python_callable=task_clean)
        t3_transform = PythonOperator(task_id="transform_and_engineer_features", python_callable=task_transform)
        t4_load = PythonOperator(task_id="load_sqlite_data_warehouse", python_callable=task_load)

        t1_extract >> t2_clean >> t3_transform >> t4_load

except ImportError:
    if __name__ == "__main__":
        print("Apache Airflow is not installed in the current environment. Running pipeline directly...")
        from src.etl_pipeline import run_pipeline
        run_pipeline()
