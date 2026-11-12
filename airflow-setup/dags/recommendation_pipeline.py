from datetime import datetime

from airflow.sdk import dag, task


PROJECT_DIR = "/opt/airflow/project"


@dag(
    dag_id="dm4ml_recommendation_pipeline",
    description="DM4ML Assignment 2 - Movie Recommendation Pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["DM4ML", "Recommendation", "Assignment2"],
)
def recommendation_pipeline():

    # Stage 1: Data Ingestion
    @task
    def ingestion():
        from src.config import ProjectPaths
        from src.ingestion.movielens import ingest_movielens

        paths = ProjectPaths.from_root(PROJECT_DIR)
        paths.create_stage1_directories()

        ingest_movielens(
            paths.raw,
            paths.reports,
        )

    # Stage 2: Data Validation
    @task
    def validation():
        from src.config import ProjectPaths
        from src.validation import validate_data

        paths = ProjectPaths.from_root(PROJECT_DIR)

        validate_data(
            paths.raw,
            paths.reports,
            paths.errors,
        )

    # Stage 3: Data Preparation
    @task
    def preparation():
        from src.config import ProjectPaths
        from src.preparation import prepare_data

        paths = ProjectPaths.from_root(PROJECT_DIR)

        prepare_data(
            paths.raw,
            paths.processed,
            rejected_records_path=(
                paths.errors / "invalid_records.csv"
            ),
        )

    # Stage 4: Data Transformation
    @task
    def transformation():
        from src.config import ProjectPaths
        from src.transformation import transform_data

        paths = ProjectPaths.from_root(PROJECT_DIR)

        transform_data(
            paths.processed,
        )

    # Stage 5: Feature Engineering
    @task
    def feature_engineering():
        from src.config import ProjectPaths
        from src.features import build_features

        paths = ProjectPaths.from_root(PROJECT_DIR)

        build_features(
            paths.processed,
            paths.features,
        )

    # Stage 6: Model Training
    @task
    def model_training():
        from src.config import ProjectPaths
        from src.training import train_models

        paths = ProjectPaths.from_root(PROJECT_DIR)

        train_models(
            paths.processed,
            paths.models,
            paths.reports,
            paths.root / "mlruns",
        )

    # Stage 7: Model Evaluation
    @task
    def model_evaluation():
        from src.config import ProjectPaths
        from src.evaluation import evaluate_models

        paths = ProjectPaths.from_root(PROJECT_DIR)

        evaluate_models(
            paths.processed,
            paths.models,
            paths.reports,
            paths.root / "mlruns",
        )

    # Define task execution order

    ingest = ingestion()
    validate = validation()
    prepare = preparation()
    transform = transformation()
    features = feature_engineering()
    train = model_training()
    evaluate = model_evaluation()

    (
        ingest
        >> validate
        >> prepare
        >> transform
        >> features
        >> train
        >> evaluate
    )


recommendation_pipeline()