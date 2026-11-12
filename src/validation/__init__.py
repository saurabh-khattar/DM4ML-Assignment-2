import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import relative_artifact_path


LOGGER = logging.getLogger(__name__)
TABLES = ("users", "movies", "ratings")
RATING_COLUMNS = {"user_id", "movie_id", "rating", "timestamp"}
RATING_MIN = 1
RATING_MAX = 5


def _read_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in TABLES:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Required ingestion output not found: {path}")
        tables[name] = pd.read_csv(path)
    if not RATING_COLUMNS.issubset(tables["ratings"].columns):
        raise ValueError(f"ratings.csv must include columns: {sorted(RATING_COLUMNS)}")
    if "user_id" not in tables["users"] or "movie_id" not in tables["movies"]:
        raise ValueError("users.csv and movies.csv must include their ID columns")
    return tables


def validate_data(
    raw_dir: str | Path,
    report_dir: str | Path,
    errors_dir: str | Path,
) -> dict[str, Any]:
    """Run quality checks and write a JSON report plus rejected source records."""
    raw_path, reports_path, errors_path = map(Path, (raw_dir, report_dir, errors_dir))
    reports_path.mkdir(parents=True, exist_ok=True)
    errors_path.mkdir(parents=True, exist_ok=True)
    tables = _read_tables(raw_path)
    failures: dict[str, dict[int, list[str]]] = {name: {} for name in TABLES}

    def mark(table_name: str, mask: pd.Series, issue: str) -> int:
        indices = tables[table_name].index[mask.fillna(False)]
        for index in indices:
            failures[table_name].setdefault(int(index), []).append(issue)
        return len(indices)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, count: int) -> None:
        checks.append({"check": name, "failure_count": int(count), "passed": count == 0})

    required_fields = {
        "users": ["user_id", "age", "gender", "occupation"],
        "movies": ["movie_id", "title"],
        "ratings": sorted(RATING_COLUMNS),
    }
    missing_count = 0
    warnings: list[dict[str, Any]] = []
    for name, frame in tables.items():
        missing_columns = [column for column in required_fields[name] if column not in frame]
        if missing_columns:
            raise ValueError(f"{name}.csv is missing required columns: {missing_columns}")
        missing_mask = frame[required_fields[name]].isna().any(axis=1)
        missing_count += mark(name, missing_mask, "missing_required_value")
    for column in ("release_date", "video_release_date", "imdb_url"):
        if column in tables["movies"]:
            missing_indices = tables["movies"].index[tables["movies"][column].isna()]
            if len(missing_indices):
                warnings.append(
                    {
                        "table_name": "movies",
                        "field": column,
                        "issue": "missing_optional_value",
                        "classification": "warning",
                        "record_count": len(missing_indices),
                        "source_rows": [int(index) for index in missing_indices],
                        "handling": "preserved and filled during preparation",
                    }
                )
    add_check("missing_values", missing_count)

    duplicate_count = 0
    duplicate_specs = {
        "users": ["user_id"],
        "movies": ["movie_id"],
        "ratings": ["user_id", "movie_id"],
    }
    for name, keys in duplicate_specs.items():
        duplicate_count += mark(
            name, tables[name].duplicated(subset=keys, keep="first"), "duplicate_record"
        )
    add_check("duplicate_records", duplicate_count)

    numeric_columns = {
        "users": {"user_id": None, "age": None},
        "movies": {"movie_id": None},
        "ratings": {
            "user_id": None,
            "movie_id": None,
            "rating": None,
            "timestamp": None,
        },
    }
    type_failure_count = 0
    for name, columns in numeric_columns.items():
        frame = tables[name]
        for column in columns:
            if column not in frame.columns:
                raise ValueError(f"{name}.csv is missing required column {column!r}")
            numeric_values = pd.to_numeric(frame[column], errors="coerce")
            bad = numeric_values.isna() | numeric_values.mod(1).ne(0)
            type_failure_count += mark(name, bad, f"incorrect_type:{column}")
    add_check("incorrect_data_types", type_failure_count)

    ratings = tables["ratings"]
    numeric_ratings = pd.to_numeric(ratings["rating"], errors="coerce")
    invalid_rating_count = mark(
        "ratings",
        numeric_ratings.lt(RATING_MIN)
        | numeric_ratings.gt(RATING_MAX)
        | numeric_ratings.isna()
        | numeric_ratings.mod(1).ne(0),
        "invalid_rating",
    )
    add_check("rating_range_1_to_5", invalid_rating_count)

    timestamps = pd.to_numeric(ratings["timestamp"], errors="coerce")
    parsed_timestamps = pd.to_datetime(timestamps, unit="s", errors="coerce", utc=True)
    invalid_timestamp_count = mark(
        "ratings",
        timestamps.isna() | timestamps.le(0) | parsed_timestamps.isna(),
        "invalid_timestamp",
    )
    add_check("valid_unix_timestamps", invalid_timestamp_count)

    valid_user_rows = ~tables["users"].index.isin(failures["users"])
    valid_movie_rows = ~tables["movies"].index.isin(failures["movies"])
    known_users = set(
        pd.to_numeric(tables["users"].loc[valid_user_rows, "user_id"], errors="coerce").dropna()
    )
    known_movies = set(
        pd.to_numeric(tables["movies"].loc[valid_movie_rows, "movie_id"], errors="coerce").dropna()
    )
    unknown_users = ~pd.to_numeric(ratings["user_id"], errors="coerce").isin(known_users)
    unknown_movies = ~pd.to_numeric(ratings["movie_id"], errors="coerce").isin(known_movies)
    referential_count = mark("ratings", unknown_users | unknown_movies, "referential_integrity")
    add_check("user_movie_referential_integrity", referential_count)

    invalid_rows = []
    for table_name, row_failures in failures.items():
        for index, issues in row_failures.items():
            record = tables[table_name].loc[index].to_dict()
            invalid_rows.append(
                {"table_name": table_name, "source_row": index, "issues": "|".join(sorted(set(issues))), **record}
            )
    invalid_path = errors_path / "invalid_records.csv"
    invalid_columns = list(
        dict.fromkeys(
            ["table_name", "source_row", "issues"]
            + [column for frame in tables.values() for column in frame.columns]
        )
    )
    pd.DataFrame(invalid_rows, columns=invalid_columns).to_csv(invalid_path, index=False)

    rejected_record_count = len(invalid_rows)
    result: dict[str, Any] = {
        "stage": "data_validation",
        "status": (
            "failed" if rejected_record_count else "passed_with_warnings" if warnings else "passed"
        ),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_counts": {name: len(frame) for name, frame in tables.items()},
        "checks": checks,
        "warnings": warnings,
        "warning_count": sum(warning["record_count"] for warning in warnings),
        "rejected_record_count": rejected_record_count,
        "invalid_records_path": relative_artifact_path(invalid_path, raw_path.parent.parent),
    }
    report_path = reports_path / "validation_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log_method = LOGGER.info if result["status"] == "passed" else LOGGER.warning
    log_method(
        "Validation %s: %d check(s), %d rejected record(s)",
        result["status"],
        len(checks),
        len(invalid_rows),
    )
    return result


__all__ = ["validate_data"]
