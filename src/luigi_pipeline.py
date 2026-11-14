"""
Luigi Workflow Orchestration for the Breast Cancer Clinical Data Pipeline.
Run with: `luigi --module src.luigi_pipeline LoadWarehouseTask --local-scheduler`
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.etl_pipeline import generate_source_feed, extract_data, clean_and_validate, transform_and_engineer, load_to_warehouse

try:
    import luigi

    class ExtractSourceTask(luigi.Task):
        def output(self):
            return luigi.LocalTarget("data/raw/breast_cancer_raw_ingest.csv")

        def run(self):
            generate_source_feed(self.output().path)

    class CleanStagingTask(luigi.Task):
        def requires(self):
            return ExtractSourceTask()

        def output(self):
            return luigi.LocalTarget("data/staging/breast_cancer_cleaned_staging.csv")

        def run(self):
            df = extract_data(self.input().path)
            clean_and_validate(df, self.output().path)

    class TransformFeaturesTask(luigi.Task):
        def requires(self):
            return CleanStagingTask()

        def output(self):
            return luigi.LocalTarget("data/processed/breast_cancer_warehouse_ready.csv")

        def run(self):
            staging_df = pd.read_csv(self.input().path)
            transform_and_engineer(staging_df, self.output().path)

    class LoadWarehouseTask(luigi.Task):
        def requires(self):
            return TransformFeaturesTask()

        def output(self):
            return luigi.LocalTarget("data/processed/clinical_data_warehouse.db")

        def run(self):
            processed_df = pd.read_csv(self.input().path)
            load_to_warehouse(processed_df, self.output().path)

except ImportError:
    if __name__ == "__main__":
        print("Luigi is not installed in this environment. Running pipeline directly...")
        from src.etl_pipeline import run_pipeline
        run_pipeline()
