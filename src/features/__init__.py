import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import relative_artifact_path


LOGGER = logging.getLogger(__name__)
MOVIE_METADATA = {"movie_id", "title", "release_date", "video_release_date", "imdb_url"}


def build_features(
    processed_dir: str | Path,
    features_dir: str | Path,
) -> dict[str, Any]:
    """Build reusable user, movie, and interaction features from training data."""
    data_path, feature_path = Path(processed_dir), Path(features_dir)
    ratings_path = data_path / "ratings_train.csv"
    users_path = data_path / "users_clean.csv"
    movies_path = data_path / "movies_clean.csv"
    for path in (ratings_path, users_path, movies_path):
        if not path.exists():
            raise FileNotFoundError(f"Run data preparation before feature creation: {path}")
    feature_path.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(ratings_path)
    users = pd.read_csv(users_path)
    movies = pd.read_csv(movies_path)
    if ratings.empty:
        raise ValueError("Training ratings are empty")
    genre_columns = [column for column in movies.columns if column not in MOVIE_METADATA]
    if not genre_columns:
        raise ValueError("Movie table has no genre indicator columns")

    user_stats = ratings.groupby("user_id").agg(
        user_average_rating=("rating", "mean"),
        user_activity_count=("rating", "size"),
        last_interaction_timestamp=("timestamp", "max"),
    ).reset_index()
    max_train_timestamp = int(ratings["timestamp"].max())
    user_stats["days_since_last_interaction"] = (
        max_train_timestamp - user_stats["last_interaction_timestamp"]
    ) / 86400

    genre_ratings = ratings.loc[ratings["rating"] >= 4, ["user_id", "movie_id"]].merge(
        movies[["movie_id", *genre_columns]], on="movie_id", how="inner"
    )
    long_genres = genre_ratings.melt(
        id_vars="user_id", value_vars=genre_columns, var_name="genre", value_name="has_genre"
    )
    liked_genres = long_genres.loc[long_genres["has_genre"] == 1]
    preferred = (
        liked_genres.groupby(["user_id", "genre"])
        .size()
        .rename("liked_rating_count")
        .reset_index()
        .sort_values(["user_id", "liked_rating_count", "genre"], ascending=[True, False, True])
        .drop_duplicates("user_id")
        .loc[:, ["user_id", "genre"]]
        .rename(columns={"genre": "preferred_genre"})
    )
    user_features = users.merge(user_stats, on="user_id", how="left").merge(
        preferred, on="user_id", how="left"
    )
    global_train_average = float(ratings["rating"].mean())
    user_features["user_average_rating"] = user_features["user_average_rating"].fillna(
        global_train_average
    )
    user_features["days_since_last_interaction"] = user_features[
        "days_since_last_interaction"
    ].fillna(-1)
    user_features["preferred_genre"] = user_features["preferred_genre"].fillna("unknown")
    user_features["user_activity_count"] = (
        user_features["user_activity_count"].fillna(0).astype("int64")
    )

    movie_stats = ratings.groupby("movie_id").agg(
        movie_popularity=("rating", "size"),
        movie_average_rating=("rating", "mean"),
    ).reset_index()
    movie_features = movies.merge(movie_stats, on="movie_id", how="left")
    movie_features["movie_popularity"] = movie_features["movie_popularity"].fillna(0).astype("int64")
    movie_features["movie_average_rating"] = movie_features["movie_average_rating"].fillna(
        global_train_average
    )

    interaction_features = ratings.sort_values(
        ["user_id", "timestamp"], kind="mergesort"
    ).copy()
    interaction_features["days_since_previous_interaction"] = (
        interaction_features.groupby("user_id")["timestamp"].diff() / 86400
    )

    outputs = {
        "user_features.csv": user_features,
        "movie_features.csv": movie_features,
        "interaction_features.csv": interaction_features,
    }
    for filename, frame in outputs.items():
        frame.to_csv(feature_path / filename, index=False)
    catalog = pd.DataFrame(
        [
            {
                "feature_name": "user_average_rating",
                "definition": "Mean explicit training rating; global training mean fallback for users without ratings",
                "data_type": "float",
                "source": "ratings_train.csv",
                "entity": "user",
            },
            {
                "feature_name": "user_activity_count",
                "definition": "Number of training ratings recorded for the user",
                "data_type": "integer",
                "source": "ratings_train.csv",
                "entity": "user",
            },
            {
                "feature_name": "preferred_genre",
                "definition": "Most frequent genre among the user's training ratings of 4 or 5",
                "data_type": "string",
                "source": "ratings_train.csv + movies_clean.csv",
                "entity": "user",
            },
            {
                "feature_name": "days_since_last_interaction",
                "definition": "Days between the user's last training rating and latest training timestamp; -1 for no training interaction",
                "data_type": "float",
                "source": "ratings_train.csv",
                "entity": "user",
            },
            {
                "feature_name": "movie_popularity",
                "definition": "Number of training ratings for the movie",
                "data_type": "integer",
                "source": "ratings_train.csv",
                "entity": "movie",
            },
            {
                "feature_name": "movie_average_rating",
                "definition": "Mean explicit training rating; global training mean fallback for movies without ratings",
                "data_type": "float",
                "source": "ratings_train.csv",
                "entity": "movie",
            },
            {
                "feature_name": "days_since_previous_interaction",
                "definition": "Elapsed days since the user's previous training interaction; first interaction is blank",
                "data_type": "float",
                "source": "ratings_train.csv",
                "entity": "interaction",
            },
        ]
    )
    catalog_path = feature_path / "feature_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    result: dict[str, Any] = {
        "stage": "feature_management",
        "status": "success",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "training split only (test data excluded)",
        "outputs": {
            **{
                name.removesuffix(".csv"): relative_artifact_path(
                    feature_path / name, feature_path.parent.parent
                )
                for name in outputs
            },
            "feature_catalog": relative_artifact_path(
                catalog_path, feature_path.parent.parent
            ),
        },
        "shapes": {name.removesuffix(".csv"): list(frame.shape) for name, frame in outputs.items()},
        "feature_count": len(catalog),
    }
    (feature_path / "feature_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Created feature datasets and catalog (%d features)", len(catalog))
    return result


__all__ = ["build_features"]
