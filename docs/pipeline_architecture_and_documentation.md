# Data Pipeline Architecture and Functionality Documentation

## 1. Executive Overview
This project implements a production-grade, modular **Extract-Transform-Load (ETL) Data Pipeline** for clinical oncology diagnostics using the **Breast Cancer Wisconsin Diagnostic Dataset**. The pipeline automates the end-to-end journey from a noisy upstream data feed (`data/raw/breast_cancer_raw_ingest.csv`) through staging validation (`data/staging/`) to a model-ready relational **SQLite Data Warehouse** (`data/processed/clinical_data_warehouse.db`) and analytical CSV store.

To support diverse deployment environments, the pipeline is engineered with **three orchestration interfaces**:
1. **Apache Airflow DAG (`dags/breast_cancer_etl_dag.py`):** Scheduled `@daily` task graph using `PythonOperator` with automatic retries (`retries=2`, `retry_delay=5m`).
2. **Luigi Task Graph (`src/luigi_pipeline.py`):** Target-driven dependency resolution (`ExtractSourceTask -> CleanStagingTask -> TransformFeaturesTask -> LoadWarehouseTask`) ensuring idempotency.
3. **Standalone CLI / Google Colab Runner (`src/etl_pipeline.py`):** Zero-overhead Python execution with structured timestamped logging and JSON audit generation.

---

## 2. Pipeline Architecture & Data Lineage

```text
[Upstream Source]       [Stage 1: Extract]         [Stage 2: Clean & Validate]
 Raw Clinical CSV  --->  extract_data()       --->  clean_and_validate()
 (584 rows, 35 cols)     Ingest & schema check      - Drop 15 duplicate patient_ids
                                                    - Median impute 27 missing cells
                                                    - Cap 612 IQR outlier values
                                                    - Assert 0 nulls & unique PKs
                                                               |
                                                               v
[Stage 4: Load Warehouse]                          [Stage 3: Transform & Engineer]
 load_to_warehouse()                          <---  transform_and_engineer()
 - SQLite: fact_breast_cancer_diagnostics           - Create 5 domain ratio features
 - SQLite: agg_diagnosis_by_size_grade              - Ordinal + One-Hot Encoding
 - Audit:  reports/pipeline_execution_audit.json    - Z-Score scale 35 numeric cols
 (569 clean rows, 79 total columns)
```

---

## 3. Functional Breakdown of Pipeline Stages

### Stage 1: Data Extraction (`extract_data`)
- **Input:** `data/raw/breast_cancer_raw_ingest.csv` (`584` rows $\times$ `35` columns).
- **Functionality:** Reads the raw clinical feed containing `30` cytological nucleus measurements, patient identifier (`patient_id`), categorical attributes (`tumor_size_grade`, `texture_pattern`, `diagnosis_label`), and binary `target`. If the upstream feed is missing in a fresh environment, `generate_source_feed()` deterministically synthesizes the raw drop.

### Stage 2: Data Cleaning & Quality Gate Validation (`clean_and_validate`)
- **Deduplication:** Identifies and removes `15` duplicate `patient_id` entries (`584 -> 569` unique patients).
- **Missing Value Imputation:** Imputes `27` missing cells across `mean smoothness` (`15` nulls) and `mean compactness` (`12` nulls) using column medians to prevent skew distortion.
- **Outlier Remediation:** Applies the $1.5 \times \text{IQR}$ clipping rule across all `30` continuous cytological features, capping `612` extreme tail values without discarding valid patient records.
- **Automated Quality Gates:** Executes hard `assert` checks verifying `0` remaining nulls and `100%` unique primary keys (`patient_id`) before writing `data/staging/breast_cancer_cleaned_staging.csv`.

### Stage 3: Feature Engineering & Transformation (`transform_and_engineer`)
- **Domain Feature Creation (`+5` features):**
  1. `area_perimeter_ratio`: $\text{mean area} / \text{mean perimeter}$ (compactness density).
  2. `radius_worst_to_mean_ratio`: $\text{worst radius} / \text{mean radius}$ (radial growth expansion).
  3. `concavity_smoothness_index`: $\text{mean concavity} \times \text{mean smoothness}$ (boundary roughness interaction).
  4. `circularity_deviation`: $|(\text{mean perimeter} / (2\pi \cdot \text{mean radius})) - 1|$ (departure from circularity).
  5. `nuclear_malformation_score`: $\text{worst concavity} + \text{worst concave points} + \text{worst compactness}$.
- **Categorical Encoding (`+4` columns):** Maps ordinal `tumor_size_grade` (`Small=0, Medium=1, Large=2`) and One-Hot Encodes nominal `texture_pattern` (`Smooth`, `Irregular`, `Rough`).
- **Standardization (`+35` scaled columns):** Applies `StandardScaler` ($\mu=0, \sigma=1$) across all `35` raw and engineered numerical features, outputting `79` total columns to `data/processed/breast_cancer_warehouse_ready.csv`.

### Stage 4: Data Warehouse Loading & Analytical Aggregation (`load_to_warehouse`)
- **Destination Database:** `data/processed/clinical_data_warehouse.db` (SQLite relational database).
- **Tables Created:**
  - `fact_breast_cancer_diagnostics` (`569` rows $\times$ `79` columns): Granular patient-level feature store for machine learning training and inference.
  - `agg_diagnosis_by_size_grade` (`6` rows $\times$ `7` columns): Pre-aggregated analytical mart summarizing patient counts and mean biomarker scores across diagnosis and tumor size tiers:

| Diagnosis Label | Tumor Size Grade | Patient Count | Avg Mean Radius | Avg Worst Radius | Avg Area/Perimeter Ratio | Avg Malformation Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | Large | 23 | 15.476 | 16.888 | 7.343 | 0.515 |
| **Benign** | Medium | 149 | 13.321 | 14.666 | 6.368 | 0.454 |
| **Benign** | Small | 185 | 10.787 | 11.907 | 5.153 | 0.384 |
| **Malignant** | Large | 167 | 18.337 | 22.178 | 8.488 | 0.979 |
| **Malignant** | Medium | 39 | 13.700 | 16.497 | 6.446 | 1.010 |
| **Malignant** | Small | 6 | 11.475 | 14.152 | 5.324 | 1.123 |

---

## 4. How to Run the Pipeline

### Option A: Standalone Python CLI (Fastest — Works Everywhere Including Google Colab)
```bash
pip install -r requirements.txt
python src/etl_pipeline.py
```

### Option B: Apache Airflow Orchestration
```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
cp dags/breast_cancer_etl_dag.py $AIRFLOW_HOME/dags/
airflow dags test breast_cancer_clinical_etl_pipeline 2026-01-01
```

### Option C: Luigi Workflow Scheduler
```bash
luigi --module src.luigi_pipeline LoadWarehouseTask --local-scheduler
```
