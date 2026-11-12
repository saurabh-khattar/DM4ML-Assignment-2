import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import relative_artifact_path


LOGGER = logging.getLogger(__name__)
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def prepare_data(
    raw_dir: str | Path,
    processed_dir: str | Path,
    test_fraction: float = 0.2,
    rejected_records_path: str | Path | None = None,
) -> dict[str, Any]:
    """Clean source tables and create a deterministic chronological rating split."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    raw_path, output_path = Path(raw_dir), Path(processed_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    users = _read_csv(raw_path / "users.csv")
    movies = _read_csv(raw_path / "movies.csv")
    ratings = _read_csv(raw_path / "ratings.csv")
    original_counts = {"users": len(users), "movies": len(movies), "ratings": len(ratings)}

    rejected_source_rows: dict[str, set[int]] = {}
    if rejected_records_path is not None:
        rejected_path = Path(rejected_records_path)
        if not rejected_path.exists():
            raise FileNotFoundError(f"Validation rejection file not found: {rejected_path}")
        rejected_records = pd.read_csv(rejected_path)
        if not {"table_name", "source_row"}.issubset(rejected_records.columns):
            raise ValueError("Validation rejection file must include table_name and source_row")
        for table_name, group in rejected_records.groupby("table_name"):
            if table_name not in {"users", "movies", "ratings"}:
                raise ValueError(f"Unexpected table in validation rejection file: {table_name}")
            rejected_source_rows[table_name] = set(
                pd.to_numeric(group["source_row"], errors="raise").astype(int)
            )
        for table_name, frame in (("users", users), ("movies", movies), ("ratings", ratings)):
            rejected = rejected_source_rows.get(table_name, set())
            if rejected and not rejected.issubset(set(frame.index)):
                raise ValueError(
                    f"Validation rejection rows for {table_name} do not match current raw data"
                )
            frame.drop(index=rejected, inplace=True)

    users = users.drop_duplicates(subset=["user_id"]).dropna(subset=["user_id"])
    movies = movies.drop_duplicates(subset=["movie_id"]).dropna(subset=["movie_id"])
    filled_missing_values: dict[str, int] = {}
    for column in ("gender", "occupation", "zip_code"):
        if column in users:
            filled_missing_values[f"users.{column}"] = int(users[column].isna().sum())
            users[column] = users[column].fillna("unknown").astype("string")
    if "age" in users:
        users["age"] = pd.to_numeric(users["age"], errors="coerce")
        filled_missing_values["users.age"] = int(users["age"].isna().sum())
        users["age"] = users["age"].fillna(users["age"].median()).astype("int64")
    for column in ("title", "release_date", "imdb_url"):
        if column in movies:
            filled_missing_values[f"movies.{column}"] = int(movies[column].isna().sum())
            movies[column] = movies[column].fillna("unknown").astype("string")
    all_missing_columns = [
        column for column in movies.columns
        if column != "movie_id" and movies[column].isna().all()
    ]
    movies = movies.drop(columns=all_missing_columns)
    genre_columns = [column for column in movies if column not in {
        "movie_id", "title", "release_date", "video_release_date", "imdb_url"
    }]
    for column in genre_columns:
        movies[column] = pd.to_numeric(movies[column], errors="coerce").fillna(0).astype("int64")

    for column in RATING_COLUMNS:
        if column not in ratings:
            raise ValueError(f"ratings.csv is missing required column {column!r}")
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
    ratings = ratings.dropna(subset=RATING_COLUMNS)
    ratings = ratings[
        ratings["rating"].between(1, 5)
        & ratings["timestamp"].gt(0)
        & ratings["user_id"].isin(users["user_id"])
        & ratings["movie_id"].isin(movies["movie_id"])
    ].copy()
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"], keep="first")
    ratings = ratings.astype(
        {"user_id": "int64", "movie_id": "int64", "rating": "int64", "timestamp": "int64"}
    )
    ratings = ratings.sort_values(["timestamp", "user_id", "movie_id"], kind="mergesort")
    if len(ratings) < 2:
        raise ValueError("At least two valid unique ratings are required for a train/test split")

    target_train_records = int(len(ratings) * (1 - test_fraction))
    target_train_records = min(max(target_train_records, 1), len(ratings) - 1)
    cutoff_timestamp = int(ratings.iloc[target_train_records]["timestamp"])
    records_before_cutoff = int((ratings["timestamp"] < cutoff_timestamp).sum())
    records_through_cutoff = int((ratings["timestamp"] <= cutoff_timestamp).sum())
    if abs(records_before_cutoff - target_train_records) <= abs(
        records_through_cutoff - target_train_records
    ):
        train = ratings.loc[ratings["timestamp"] < cutoff_timestamp].copy()
        test = ratings.loc[ratings["timestamp"] >= cutoff_timestamp].copy()
        cutoff_group_assignment = "test"
    else:
        train = ratings.loc[ratings["timestamp"] <= cutoff_timestamp].copy()
        test = ratings.loc[ratings["timestamp"] > cutoff_timestamp].copy()
        cutoff_group_assignment = "train"
    if train.empty or test.empty:
        raise ValueError("Timestamp-grouped chronological split produced an empty partition")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("Chronological split invariant failed: timestamp groups overlap")

    outputs = {
        "users_clean.csv": users,
        "movies_clean.csv": movies,
        "ratings_clean.csv": ratings,
        "ratings_train.csv": train,
        "ratings_test.csv": test,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_path / filename, index=False)

    result: dict[str, Any] = {
        "stage": "data_preparation",
        "status": "success",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_record_counts": original_counts,
        "clean_record_counts": {"users": len(users), "movies": len(movies), "ratings": len(ratings)},
        "removed_record_counts": {
            name: original_counts[name] - len(frame)
            for name, frame in (("users", users), ("movies", movies), ("ratings", ratings))
        },
        "rejected_validation_record_counts": {
            name: len(rejected_source_rows.get(name, set()))
            for name in ("users", "movies", "ratings")
        },
        "filled_missing_values": filled_missing_values,
        "dropped_all_missing_movie_columns": all_missing_columns,
        "split": {
            "method": "chronological; all interactions with the cutoff timestamp remain together",
            "requested_test_fraction": test_fraction,
            "target_train_records": target_train_records,
            "train_records": len(train),
            "test_records": len(test),
            "train_fraction": len(train) / len(ratings),
            "test_fraction": len(test) / len(ratings),
            "train_max_timestamp": int(train["timestamp"].max()),
            "test_min_timestamp": int(test["timestamp"].min()),
            "cutoff_timestamp": cutoff_timestamp,
            "cutoff_group_assignment": cutoff_group_assignment,
            "cutoff_timestamp_in_train": bool((train["timestamp"] == cutoff_timestamp).any()),
            "cutoff_timestamp_in_test": bool((test["timestamp"] == cutoff_timestamp).any()),
        },
        "outputs": [
            relative_artifact_path(output_path / filename, raw_path.parent.parent)
            for filename in outputs
        ],
    }
    (output_path / "preparation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Prepared %d ratings: %d train, %d test",
        len(ratings),
        len(train),
        len(test),
    )
    return result


__all__ = ["prepare_data"]
