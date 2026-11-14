# Building a Data Pipeline

End-to-end automated **Extract-Transform-Load (ETL) Data Pipeline** with **Apache Airflow**, **Luigi**, and **Standalone CLI** orchestration on the Breast Cancer Wisconsin Diagnostic Dataset.

## Repository Structure
```text
├── dags/
│   └── breast_cancer_etl_dag.py                        # Apache Airflow DAG definition (@daily schedule)
├── src/
│   ├── etl_pipeline.py                                 # Core modular ETL pipeline & CLI runner
│   └── luigi_pipeline.py                               # Luigi task-dependency workflow definition
├── data/
│   ├── raw/
│   │   └── breast_cancer_raw_ingest.csv                # Raw source feed (584 rows, duplicates & nulls)
│   ├── staging/
│   │   └── breast_cancer_cleaned_staging.csv           # Deduplicated, imputed & IQR-capped staging table
│   └── processed/
│       ├── breast_cancer_warehouse_ready.csv           # Model-ready feature store (569 rows × 79 cols)
│       └── clinical_data_warehouse.db                  # SQLite relational data warehouse
├── docs/
│   └── pipeline_architecture_and_documentation.md      # Full architecture & functional documentation
├── reports/
│   ├── pipeline_execution_audit.json                   # Automated run audit metrics
│   ├── warehouse_summary_query.csv                     # SQL aggregation output table
│   ├── pipeline_architecture_diagram.png               # Visual pipeline lineage diagram
│   └── pipeline_audit_and_warehouse_metrics.png        # Data quality & warehouse query chart
├── notebooks/
│   └── data_pipeline_walkthrough.ipynb                 # Interactive Google Colab / Jupyter walkthrough
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt
python src/etl_pipeline.py
```
