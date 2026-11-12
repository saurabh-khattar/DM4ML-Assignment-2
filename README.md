# DM4ML Assignment 2: Movie Recommendation Pipeline

This project implements the data pipeline through model evaluation for the
MovieLens 100K ratings dataset. It uses simple Python modules and local CSV,
pickle, and SQLite-backed MLflow storage. Workflow orchestration is intentionally
out of scope.

## Setup

Use **64-bit Python 3.11** for both macOS Apple Silicon development and a future
Airflow Linux/ARM64 container. It is supported by Airflow, MLflow, pandas,
NumPy, and the published ARM64 `scikit-surprise` wheel.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`scikit-surprise==1.1.5` is pinned because it provides CPython 3.11 wheels for
macOS ARM64 and Linux ARM64 and supports the NumPy 2 line. If pip cannot use a
wheel for a different platform or Python version, it may need to build the
extension from source; install a C/C++ compiler and Python development headers
in that environment. Airflow 3.4.0's Python 3.11 constraints currently specify
NumPy 2.4.6 and pandas 3.1.0, which fit the project requirement ranges.
`cffi==2.1.1`, a transitive dependency selected by those constraints, has no
matching CPython 3.11 ARM64 wheel in the checked package index; its source
distribution is available. If pip must build it, provide a C compiler and
libffi development headers. Airflow is intentionally **not** included in
`requirements.txt`.

## Run the complete pipeline

Run from any terminal directory:

```bash
python /path/to/Assignment\ 2/main.py
```

By default, the project root is resolved from `main.py`, not the current
working directory. To use another output directory, pass `--project-dir
/path/to/project-data`. The pipeline downloads MovieLens 100K if
`data/raw/ml-100k.zip` is not present, then executes ingestion, validation,
preparation, transformation, feature creation, training, and evaluation.

## Notebook

Open `DM4ML_Assignment_2.ipynb` and use **Restart Kernel → Run All** to inspect
actual stage reports and result tables. The notebook finds the project through
its working directory or Python path. If it is not discoverable, set the
`DM4ML_PROJECT_ROOT` environment variable to the project directory before
starting the notebook kernel. The notebook reuses saved data and models where
available.

## Project outputs

- `data/raw/`: original MovieLens archive and ingested tables.
- `data/errors/invalid_records.csv`: rows rejected by validation, with issue
  labels.
- `data/processed/`: cleaned entities, chronological 80/20 ratings split,
  user activity, movie popularity, interaction matrix, and stage reports.
- `data/features/`: user, movie, and interaction feature tables and catalog.
- `models/`: serialized popularity baseline and Surprise SVD model.
- `reports/`: ingestion, validation, preparation, training, and evaluation
  reports plus the model comparison table.
- `mlruns/`: local SQLite MLflow tracking database and model artifacts.

MLflow uses `mlruns/mlflow.db` under the project root by default. To use a
separate database or a tracking server (for example, in a container), set
`MLFLOW_TRACKING_URI` before running the pipeline. A new SQLite database is
created at the configured location when it does not exist; existing tracking
databases are left intact. For a container, point the URI at a writable
persistent volume, for example:

```bash
export MLFLOW_TRACKING_URI='sqlite:////path/to/persistent/mlruns/mlflow.db'
```

The tracking URI is not written to reports; local SQLite database references
are recorded relative to the project root when possible.

The chronological split sorts interactions by Unix timestamp, assigning the
earliest 80% to training and latest 20% to test. Derived interaction
statistics, features, and model fitting use training data only. Evaluation
uses users with at least one held-out rating of 4 or 5. Precision@10 divides
hits by 10; Recall@10 and NDCG@10 use held-out relevant items. Coverage is the
fraction of the movie catalog recommended, and diversity is mean pairwise
genre Jaccard distance. RMSE and MAE are calculated for each model on every
held-out explicit rating.

## Dependency and future Docker notes

The pipeline imports pandas, NumPy, Surprise, and MLflow; it does not import
scikit-learn, so scikit-learn is not a direct project requirement. For a future
container, use the official multi-platform Airflow image on Linux/ARM64, such
as a versioned `apache/airflow:<airflow-version>-python3.11` image. Install
Airflow using its matching official constraints file for the exact Airflow and
Python versions, then add project dependencies without overriding those
constraints. Keep Airflow dependencies separate from this project's
`requirements.txt`. No Airflow or Docker image has been installed or tested as
part of this project.

Compatibility references:

- [Airflow supported versions and constraints](https://airflow.apache.org/docs/apache-airflow/stable/installation/index.html)
- [Official Airflow Docker image platforms](https://airflow.apache.org/docs/docker-stack/index.html)
- [scikit-surprise releases and platform wheels](https://pypi.org/project/scikit-surprise/)
- [NumPy 2.3 Python support](https://numpy.org/doc/2.3/release/2.3.0-notes.html)
- [pandas releases and Python support](https://pypi.org/project/pandas/)
- [MLflow installation](https://mlflow.org/docs/latest/ml/getting-started/installing-mlflow/)

## Module entry points

- `src.ingestion.movielens.ingest_movielens(raw_dir, report_dir)`
- `src.validation.validate_data(raw_dir, report_dir, errors_dir)`
- `src.preparation.prepare_data(raw_dir, processed_dir, test_fraction=0.2)`
- `src.transformation.transform_data(processed_dir)`
- `src.features.build_features(processed_dir, features_dir)`
- `src.training.train_models(processed_dir, models_dir, reports_dir, tracking_dir)`
- `src.evaluation.evaluate_models(processed_dir, models_dir, reports_dir, tracking_dir, k=10)`
