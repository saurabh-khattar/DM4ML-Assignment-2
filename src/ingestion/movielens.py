import json
import logging
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import relative_artifact_path


LOGGER = logging.getLogger(__name__)
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
GENRE_COLUMNS = [
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]
MOVIE_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    *GENRE_COLUMNS,
]
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


def _download_archive(archive_path: Path, dataset_url: str) -> None:
    if archive_path.exists():
        LOGGER.info("Using existing dataset archive: %s", archive_path)
        return

    temporary_path = archive_path.with_name(archive_path.name + ".part")
    LOGGER.info("Downloading MovieLens 100K from %s", dataset_url)
    try:
        with urllib.request.urlopen(dataset_url, timeout=60) as response:
            with temporary_path.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file)
        temporary_path.replace(archive_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_dataset_tables(archive_path: Path) -> dict[str, pd.DataFrame]:
    members = {
        "users": "ml-100k/u.user",
        "movies": "ml-100k/u.item",
        "ratings": "ml-100k/u.data",
    }
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(archive_path) as archive:
        missing_members = [name for name in members.values() if name not in archive.namelist()]
        if missing_members:
            raise ValueError(f"Dataset archive is missing expected files: {missing_members}")

        tables["users"] = pd.read_csv(
            archive.open(members["users"]),
            sep="|",
            header=None,
            names=USER_COLUMNS,
            dtype={
                "user_id": "int64",
                "age": "int64",
                "gender": "string",
                "occupation": "string",
                "zip_code": "string",
            },
            keep_default_na=False,
        )
        movie_dtypes = {"movie_id": "int64"}
        movie_dtypes.update({genre: "int64" for genre in GENRE_COLUMNS})
        movie_dtypes.update(
            {
                "title": "string",
                "release_date": "string",
                "video_release_date": "string",
                "imdb_url": "string",
            }
        )
        tables["movies"] = pd.read_csv(
            archive.open(members["movies"]),
            sep="|",
            header=None,
            names=MOVIE_COLUMNS,
            encoding="latin-1",
            dtype=movie_dtypes,
            keep_default_na=False,
        )
        tables["ratings"] = pd.read_csv(
            archive.open(members["ratings"]),
            sep="\t",
            header=None,
            names=RATING_COLUMNS,
            dtype={
                "user_id": "int64",
                "movie_id": "int64",
                "rating": "int64",
                "timestamp": "int64",
            },
        )
    return tables


def ingest_movielens(
    raw_dir: str | Path,
    report_dir: str | Path,
    dataset_url: str = DATASET_URL,
) -> dict[str, Any]:
    """Download and save standardized, otherwise unmodified MovieLens tables."""
    raw_path = Path(raw_dir)
    reports_path = Path(report_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    archive_path = raw_path / "ml-100k.zip"
    report_path = reports_path / "ingestion_report.json"
    report: dict[str, Any] = {
        "stage": "data_ingestion",
        "source": dataset_url,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        _download_archive(archive_path, dataset_url)
        tables = _read_dataset_tables(archive_path)

        outputs: dict[str, str] = {}
        record_counts: dict[str, int] = {}
        for name, table in tables.items():
            output_path = raw_path / f"{name}.csv"
            table.to_csv(output_path, index=False)
            outputs[name] = relative_artifact_path(output_path, raw_path.parent.parent)
            record_counts[name] = len(table)
            LOGGER.info("Ingested %s records from %s", len(table), name)

        report.update(
            {
                "status": "success",
                "archive": relative_artifact_path(archive_path, raw_path.parent.parent),
                "record_counts": record_counts,
                "outputs": outputs,
            }
        )
        LOGGER.info("MovieLens ingestion succeeded")
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        LOGGER.exception("MovieLens ingestion failed")
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report
