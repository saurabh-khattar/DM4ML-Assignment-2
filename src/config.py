from dataclasses import dataclass
import os
from pathlib import Path


def relative_artifact_path(path: str | Path, project_root: str | Path) -> str:
    """Return an artifact path relative to the project root."""
    resolved_path = Path(path).resolve()
    resolved_root = Path(project_root).resolve()
    return Path(os.path.relpath(resolved_path, resolved_root)).as_posix()


def resolve_mlflow_tracking(tracking_dir: str | Path) -> tuple[str, Path | None]:
    """Resolve the configured MLflow URI and prepare a local SQLite database path."""
    from sqlalchemy.engine import make_url

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = f"sqlite:///{(Path(tracking_dir) / 'mlflow.db').resolve()}"

    parsed_uri = make_url(tracking_uri)
    if (
        parsed_uri.get_backend_name() != "sqlite"
        or not parsed_uri.database
        or parsed_uri.database == ":memory:"
    ):
        return tracking_uri, None

    database_path = Path(parsed_uri.database).expanduser()
    if not database_path.is_absolute():
        database_path = Path(tracking_dir) / database_path
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return parsed_uri.set(database=str(database_path)).render_as_string(), database_path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        return cls(root=Path(root).expanduser().resolve())

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features"

    @property
    def errors(self) -> Path:
        return self.root / "data" / "errors"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def create_stage1_directories(self) -> None:
        for directory in (
            self.raw,
            self.processed,
            self.features,
            self.errors,
            self.reports,
            self.models,
        ):
            directory.mkdir(parents=True, exist_ok=True)
