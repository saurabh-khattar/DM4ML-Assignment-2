import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import relative_artifact_path, resolve_mlflow_tracking


LOGGER = logging.getLogger(__name__)
EXPERIMENT_NAME = "DM4ML-MovieLens100K"


def train_models(
    processed_dir: str | Path,
    models_dir: str | Path,
    reports_dir: str | Path,
    tracking_dir: str | Path,
) -> dict[str, Any]:
    """Train a popularity baseline and Surprise SVD, logging both with MLflow."""
    try:
        import mlflow
        from surprise import Dataset, Reader, SVD
    except ImportError as error:
        raise ImportError(
            "Model training requires dependencies in requirements.txt: mlflow and scikit-surprise"
        ) from error

    data_path, model_path, report_path, tracking_path = map(
        Path, (processed_dir, models_dir, reports_dir, tracking_dir)
    )
    train_csv = data_path / "ratings_train.csv"
    for path in (train_csv, data_path / "movies_clean.csv"):
        if not path.exists():
            raise FileNotFoundError(f"Run preparation before model training: {path}")
    model_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(train_csv)
    if ratings.empty:
        raise ValueError("Training ratings are empty")
    popularity = (
        ratings.groupby("movie_id")
        .agg(rating_count=("rating", "size"), average_rating=("rating", "mean"))
        .reset_index()
        .sort_values(["rating_count", "average_rating", "movie_id"], ascending=[False, False, True])
    )
    popularity_model = {
        "movie_ids": popularity["movie_id"].astype(int).tolist(),
        "rating_count": popularity["rating_count"].astype(int).tolist(),
        "average_rating": popularity["average_rating"].astype(float).tolist(),
        "global_average_rating": float(ratings["rating"].mean()),
    }
    popularity_artifact = model_path / "popularity_model.pkl"
    with popularity_artifact.open("wb") as file:
        pickle.dump(popularity_model, file)

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)
    svd_parameters = {"n_factors": 50, "n_epochs": 20, "random_state": 42, "verbose": False}
    svd_model = SVD(**svd_parameters)
    svd_model.fit(dataset.build_full_trainset())
    svd_artifact = model_path / "svd_model.pkl"
    with svd_artifact.open("wb") as file:
        pickle.dump(svd_model, file)

    tracking_uri, tracking_database = resolve_mlflow_tracking(tracking_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs = []
    model_specs = [
        (
            "popularity",
            popularity_artifact,
            {"algorithm": "rating-count ranking", "train_records": len(ratings)},
        ),
        (
            "svd",
            svd_artifact,
            {"algorithm": "Surprise SVD", "train_records": len(ratings), **svd_parameters},
        ),
    ]
    for model_name, artifact_path, parameters in model_specs:
        with mlflow.start_run(run_name=model_name) as active_run:
            mlflow.log_params(parameters)
            mlflow.log_metric("training_records", len(ratings))
            mlflow.log_artifact(str(artifact_path), artifact_path="models")
            runs.append(
                {
                    "model": model_name,
                    "run_id": active_run.info.run_id,
                    "artifact": relative_artifact_path(artifact_path, model_path.parent),
                    "parameters": parameters,
                }
            )

    report: dict[str, Any] = {
        "stage": "model_training",
        "status": "success",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_records": len(ratings),
        "experiment_name": EXPERIMENT_NAME,
        "tracking_database": (
            relative_artifact_path(tracking_database, model_path.parent)
            if tracking_database
            else None
        ),
        "tracking_uri_source": (
            "MLFLOW_TRACKING_URI" if os.environ.get("MLFLOW_TRACKING_URI") else "project_default"
        ),
        "models": runs,
    }
    (report_path / "model_training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Trained popularity and SVD models; MLflow runs: %s", [run["run_id"] for run in runs])
    return report


__all__ = ["train_models"]
