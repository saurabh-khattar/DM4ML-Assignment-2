import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import relative_artifact_path


LOGGER = logging.getLogger(__name__)


def transform_data(processed_dir: str | Path) -> dict[str, Any]:
    """Create training-only user activity, movie popularity, and interaction matrix."""
    data_path = Path(processed_dir)
    ratings_path = data_path / "ratings_train.csv"
    users_path = data_path / "users_clean.csv"
    movies_path = data_path / "movies_clean.csv"
    for path in (ratings_path, users_path, movies_path):
        if not path.exists():
            raise FileNotFoundError(f"Run data preparation before transformation: {path}")

    ratings = pd.read_csv(ratings_path)
    users = pd.read_csv(users_path)
    movies = pd.read_csv(movies_path)
    if ratings.empty:
        raise ValueError("Training ratings are empty")

    user_activity = ratings.groupby("user_id").agg(
        activity_count=("rating", "size"),
        user_average_rating=("rating", "mean"),
        last_interaction_timestamp=("timestamp", "max"),
    ).reset_index()
    movie_popularity = ratings.groupby("movie_id").agg(
        popularity_count=("rating", "size"),
        movie_average_rating=("rating", "mean"),
    ).reset_index()
    movie_popularity = movies[["movie_id", "title"]].merge(
        movie_popularity, on="movie_id", how="left"
    )
    movie_popularity["popularity_count"] = (
        movie_popularity["popularity_count"].fillna(0).astype("int64")
    )

    matrix = ratings.pivot(index="user_id", columns="movie_id", values="rating")
    matrix = matrix.reindex(index=users["user_id"], columns=movies["movie_id"])
    matrix.index.name = "user_id"
    matrix.columns.name = "movie_id"

    activity_path = data_path / "user_activity.csv"
    popularity_path = data_path / "movie_popularity.csv"
    matrix_path = data_path / "interaction_matrix.csv"
    user_activity.to_csv(activity_path, index=False)
    movie_popularity.to_csv(popularity_path, index=False)
    matrix.fillna(0).to_csv(matrix_path)

    result: dict[str, Any] = {
        "stage": "data_transformation",
        "status": "success",
        "transformed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ratings_train.csv only (test data excluded)",
        "outputs": {
            "user_activity": relative_artifact_path(activity_path, data_path.parent.parent),
            "movie_popularity": relative_artifact_path(popularity_path, data_path.parent.parent),
            "interaction_matrix": relative_artifact_path(matrix_path, data_path.parent.parent),
        },
        "shapes": {
            "user_activity": list(user_activity.shape),
            "movie_popularity": list(movie_popularity.shape),
            "interaction_matrix": list(matrix.shape),
        },
        "interaction_matrix_missing_value_encoding": "0 means no training interaction",
    }
    (data_path / "transformation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Created user activity (%d), popularity (%d), matrix %s",
        len(user_activity),
        len(movie_popularity),
        matrix.shape,
    )
    return result


__all__ = ["transform_data"]
