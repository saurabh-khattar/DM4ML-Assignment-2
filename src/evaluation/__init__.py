import json
import logging
import math
import pickle
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import relative_artifact_path, resolve_mlflow_tracking


LOGGER = logging.getLogger(__name__)
K = 10
RELEVANT_RATING = 4


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required model training report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ndcg(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, movie_id in enumerate(recommended[:k])
        if movie_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _diversity(recommended: list[int], movie_genres: dict[int, set[str]]) -> float:
    pairs = list(combinations(recommended, 2))
    if not pairs:
        return 0.0
    distances = []
    for left, right in pairs:
        left_genres, right_genres = movie_genres.get(left, set()), movie_genres.get(right, set())
        union = left_genres | right_genres
        similarity = len(left_genres & right_genres) / len(union) if union else 0.0
        distances.append(1.0 - similarity)
    return sum(distances) / len(distances)


def evaluate_models(
    processed_dir: str | Path,
    models_dir: str | Path,
    reports_dir: str | Path,
    tracking_dir: str | Path,
    k: int = K,
) -> dict[str, Any]:
    """Evaluate saved recommenders on held-out ratings with shared top-K rules."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    try:
        import mlflow
    except ImportError as error:
        raise ImportError("Model evaluation logging requires mlflow from requirements.txt") from error

    data_path, model_path, report_path, tracking_path = map(
        Path, (processed_dir, models_dir, reports_dir, tracking_dir)
    )
    train_csv, test_csv = data_path / "ratings_train.csv", data_path / "ratings_test.csv"
    movie_csv = data_path / "movies_clean.csv"
    for path in (train_csv, test_csv, movie_csv):
        if not path.exists():
            raise FileNotFoundError(f"Run data preparation before model evaluation: {path}")
    training_report = _read_json(report_path / "model_training_report.json")
    popularity_path = model_path / "popularity_model.pkl"
    svd_path = model_path / "svd_model.pkl"
    for path in (popularity_path, svd_path):
        if not path.exists():
            raise FileNotFoundError(f"Run model training before evaluation: {path}")

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    movies = pd.read_csv(movie_csv)
    if test.empty:
        raise ValueError("Test ratings are empty")
    with popularity_path.open("rb") as file:
        popularity_model = pickle.load(file)
    with svd_path.open("rb") as file:
        svd_model = pickle.load(file)

    catalog_ids = movies["movie_id"].astype(int).tolist()
    seen_by_user = train.groupby("user_id")["movie_id"].apply(
        lambda values: set(values.astype(int))
    ).to_dict()
    relevant_by_user = (
        test.loc[test["rating"] >= RELEVANT_RATING]
        .groupby("user_id")["movie_id"]
        .apply(lambda values: set(values.astype(int)))
        .to_dict()
    )
    evaluation_users = sorted(relevant_by_user)
    if not evaluation_users:
        raise ValueError(f"No users have held-out ratings >= {RELEVANT_RATING}")

    popularity_order = popularity_model["movie_ids"]
    popularity_rank = {int(movie_id): rank for rank, movie_id in enumerate(popularity_order)}
    popularity_order = sorted(catalog_ids, key=lambda movie_id: (popularity_rank.get(movie_id, len(popularity_rank)), movie_id))
    genre_columns = [
        column for column in movies.columns
        if column not in {"movie_id", "title", "release_date", "video_release_date", "imdb_url"}
    ]
    movie_genres = {
        int(row.movie_id): {
            genre for genre in genre_columns if int(getattr(row, genre)) == 1
        }
        for row in movies.itertuples(index=False)
    }

    recommendations: dict[str, dict[int, list[int]]] = {"popularity": {}, "svd": {}}
    for user_id in evaluation_users:
        candidates = [movie_id for movie_id in catalog_ids if movie_id not in seen_by_user.get(user_id, set())]
        if len(candidates) < k:
            raise ValueError(f"User {user_id} has fewer than {k} unseen catalog items")
        recommendations["popularity"][int(user_id)] = [
            movie_id for movie_id in popularity_order
            if movie_id not in seen_by_user.get(user_id, set())
        ][:k]
        scored = [
            (int(movie_id), float(svd_model.predict(int(user_id), int(movie_id)).est))
            for movie_id in candidates
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        recommendations["svd"][int(user_id)] = [movie_id for movie_id, _ in scored[:k]]

    metric_results: dict[str, dict[str, float]] = {}
    for model_name, user_recommendations in recommendations.items():
        precision_values, recall_values, ndcg_values, diversity_values = [], [], [], []
        recommended_catalog: set[int] = set()
        for user_id in evaluation_users:
            recs = user_recommendations[user_id]
            relevant = relevant_by_user[user_id]
            hits = len(set(recs) & relevant)
            precision_values.append(hits / k)
            recall_values.append(hits / len(relevant))
            ndcg_values.append(_ndcg(recs, relevant, k))
            diversity_values.append(_diversity(recs, movie_genres))
            recommended_catalog.update(recs)
        metric_results[model_name] = {
            f"precision_at_{k}": sum(precision_values) / len(precision_values),
            f"recall_at_{k}": sum(recall_values) / len(recall_values),
            f"ndcg_at_{k}": sum(ndcg_values) / len(ndcg_values),
            "coverage": len(recommended_catalog) / len(catalog_ids),
            "diversity": sum(diversity_values) / len(diversity_values),
        }

    heldout_predictions = {
        "popularity": [],
        "svd": [],
    }
    average_by_movie = train.groupby("movie_id")["rating"].mean().to_dict()
    global_average = float(popularity_model["global_average_rating"])
    for row in test.itertuples(index=False):
        user_id, movie_id, actual = int(row.user_id), int(row.movie_id), float(row.rating)
        heldout_predictions["popularity"].append(
            (actual, float(average_by_movie.get(movie_id, global_average)))
        )
        heldout_predictions["svd"].append(
            (actual, float(svd_model.predict(user_id, movie_id).est))
        )
    for model_name, pairs in heldout_predictions.items():
        errors = [prediction - actual for actual, prediction in pairs]
        metric_results[model_name]["rmse"] = math.sqrt(
            sum(error * error for error in errors) / len(errors)
        )
        metric_results[model_name]["mae"] = sum(abs(error) for error in errors) / len(errors)

    tracking_uri, _ = resolve_mlflow_tracking(tracking_path)
    mlflow.set_tracking_uri(tracking_uri)
    model_runs = {run["model"]: run["run_id"] for run in training_report["models"]}
    logged_metrics = {
        "popularity": metric_results["popularity"],
        "svd": metric_results["svd"],
    }
    for model_name, metrics in logged_metrics.items():
        run_id = model_runs.get(model_name)
        if not run_id:
            raise ValueError(f"Training report has no MLflow run ID for {model_name}")
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)

    rows = []
    for model_name, label in (
        ("popularity", "Popularity baseline"),
        ("svd", "SVD collaborative filtering"),
    ):
        rows.append({"model": label, **metric_results[model_name]})
    comparison = pd.DataFrame(rows)
    comparison_path = report_path / "evaluation_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    report: dict[str, Any] = {
        "stage": "model_evaluation",
        "status": "success",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "relevant_rating_threshold": RELEVANT_RATING,
        "evaluation_user_count": len(evaluation_users),
        "evaluation_user_rule": "users with at least one held-out rating >= 4",
        "candidate_rule": "all catalog movies not rated by the user in training data",
        "coverage_denominator": len(catalog_ids),
        "diversity_definition": "mean pairwise genre Jaccard distance within recommended lists",
        "rating_metrics_rule": "RMSE/MAE over every held-out explicit rating; popularity predicts training movie mean or global mean",
        "metrics": metric_results,
        "comparison_path": relative_artifact_path(comparison_path, report_path.parent),
        "mlflow_run_ids": model_runs,
    }
    (report_path / "evaluation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Evaluated both models at K=%d; results saved to %s", k, comparison_path)
    return report


__all__ = ["evaluate_models"]
