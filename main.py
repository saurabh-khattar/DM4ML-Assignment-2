import argparse
import logging
from pathlib import Path

from src.config import ProjectPaths
from src.evaluation import evaluate_models
from src.features import build_features
from src.ingestion.movielens import ingest_movielens
from src.preparation import prepare_data
from src.training import train_models
from src.transformation import transform_data
from src.validation import validate_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MovieLens recommendation data pipeline through evaluation."
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory in which data and report outputs are stored (default: project directory).",
    )
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.project_dir)
    paths.create_stage1_directories()

    logging.basicConfig(
        filename=paths.reports / "pipeline.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    ingest_movielens(paths.raw, paths.reports)
    validation_report = validate_data(paths.raw, paths.reports, paths.errors)
    logging.info(
        "Validation status=%s, warnings=%d, rejected_records=%d",
        validation_report["status"],
        validation_report["warning_count"],
        validation_report["rejected_record_count"],
    )
    prepare_data(
        paths.raw,
        paths.processed,
        rejected_records_path=paths.errors / "invalid_records.csv",
    )
    transform_data(paths.processed)
    build_features(paths.processed, paths.features)
    train_models(paths.processed, paths.models, paths.reports, paths.root / "mlruns")
    evaluation = evaluate_models(
        paths.processed, paths.models, paths.reports, paths.root / "mlruns"
    )
    logging.info("Pipeline completed; evaluation metrics: %s", evaluation["metrics"])


if __name__ == "__main__":
    main()
