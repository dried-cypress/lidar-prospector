from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import typer
from rich.console import Console

from prospector.config import AppConfig
from prospector.providers.http import HttpClient
from prospector.providers.lidar import LidarProvider
from prospector.training.db import connect, ensure_schema, register_model, register_training_manifest
from prospector.training.model import train_model
from prospector.training.pipeline import build_dataset, ingest_he_grid, _grid

training_app = typer.Typer(
    name="train",
    help="Build and train Prospector's local archaeological learning corpus.",
    no_args_is_help=True,
)
console = Console()


def _database_url(value: str | None) -> str:
    import os
    return value or os.getenv("PROSPECTOR_DATABASE_URL") or os.getenv("DATABASE_URL") or (
        "postgresql://prospector:prospector@localhost:5432/prospector"
    )


def _require_bbox(xmin: float | None, ymin: float | None, xmax: float | None, ymax: float | None) -> tuple[float, float, float, float]:
    values = (xmin, ymin, xmax, ymax)
    if any(value is None for value in values):
        raise typer.BadParameter("xmin, ymin, xmax and ymax are all required")
    bbox = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if not bbox[2] > bbox[0] or not bbox[3] > bbox[1]:
        raise typer.BadParameter("bbox must have positive width and height")
    return bbox


@training_app.command("init")
def train_init(
    project: Path = typer.Option(Path("./project"), "--project", "-p"),
    database_url: str | None = typer.Option(None, "--database-url", envvar="PROSPECTOR_DATABASE_URL", show_default=False),
) -> None:
    """Create the training directories and PostGIS schema."""
    config = AppConfig.for_project(project)
    config.ensure_directories()
    for path in (project / "training" / "datasets", project / "training" / "lidar", project / "models"):
        path.mkdir(parents=True, exist_ok=True)
    ensure_schema(_database_url(database_url))
    console.print(f"[green]Training stack initialised:[/green] {project.resolve()}")


@training_app.command("ingest-he")
def train_ingest_he(
    xmin: float | None = typer.Option(None, "--xmin"),
    ymin: float | None = typer.Option(None, "--ymin"),
    xmax: float | None = typer.Option(None, "--xmax"),
    ymax: float | None = typer.Option(None, "--ymax"),
    tile_size: float = typer.Option(25_000.0, "--tile-size", min=1.0, help="HE query chunk size in metres."),
    max_tiles: int | None = typer.Option(None, "--max-tiles", min=1),
    project: Path = typer.Option(Path("./project"), "--project", "-p"),
    database_url: str | None = typer.Option(None, "--database-url", envvar="PROSPECTOR_DATABASE_URL", show_default=False),
) -> None:
    """Ingest Detailed_Mapping, Monument_Extents and Project_Area in spatial chunks."""
    bbox = _require_bbox(xmin, ymin, xmax, ymax)
    config = AppConfig.for_project(project)
    config.ensure_directories()
    db_url = _database_url(database_url)
    ensure_schema(db_url)
    client = HttpClient(config.cache_dir, config.request_timeout_seconds)
    summary = ingest_he_grid(
        bbox,
        tile_size_m=tile_size,
        client=client,
        database_url=db_url,
        max_tiles=max_tiles,
    )
    console.print(f"[green]HE ingestion complete:[/green] {summary.tiles} tile(s), {summary.records} new/updated records")
    if summary.errors:
        console.print(f"[yellow]Warnings:[/yellow] {len(summary.errors)}")
        for error in summary.errors[:20]:
            console.print(f"  • {error}")


@training_app.command("build-dataset")
def train_build_dataset(
    dtm_dir: Path | None = typer.Option(None, "--dtm-dir", help="Directory of local EPSG:27700 DTM GeoTIFFs."),
    download_lidar: bool = typer.Option(False, "--download-lidar", help="Download DTM tiles with the existing NLS/EA WCS provider."),
    xmin: float | None = typer.Option(None, "--xmin"),
    ymin: float | None = typer.Option(None, "--ymin"),
    xmax: float | None = typer.Option(None, "--xmax"),
    ymax: float | None = typer.Option(None, "--ymax"),
    tile_size: float = typer.Option(1000.0, "--tile-size", min=100.0),
    max_tiles: int | None = typer.Option(None, "--max-tiles", min=1),
    max_features: int | None = typer.Option(None, "--max-features", min=1),
    negative_multiplier: int = typer.Option(2, "--negative-multiplier", min=1, max=10),
    project: Path = typer.Option(Path("./project"), "--project", "-p"),
    database_url: str | None = typer.Option(None, "--database-url", envvar="PROSPECTOR_DATABASE_URL", show_default=False),
) -> None:
    """Create rotation/scale-augmented LiDAR training examples from HE archaeology."""
    config = AppConfig.for_project(project)
    config.ensure_directories()
    db_url = _database_url(database_url)
    ensure_schema(db_url)
    raster_tiles: list[Path] = []
    if dtm_dir is not None:
        raster_tiles = sorted(dtm_dir.glob("*.tif")) + sorted(dtm_dir.glob("*.tiff"))
        if max_tiles is not None:
            raster_tiles = raster_tiles[:max_tiles]
    elif download_lidar:
        bbox = _require_bbox(xmin, ymin, xmax, ymax)
        client = HttpClient(config.cache_dir, config.request_timeout_seconds)
        provider = LidarProvider(client)
        lidar_dir = project / "training" / "lidar"
        lidar_dir.mkdir(parents=True, exist_ok=True)
        for index, tile in enumerate(_grid(bbox, tile_size), 1):
            if max_tiles is not None and index > max_tiles:
                break
            name = f"tile_{tile[0]:.0f}_{tile[1]:.0f}_{tile[2]:.0f}_{tile[3]:.0f}.tif"
            destination = lidar_dir / name
            try:
                path, _cache = provider.download_geotiff(tile, destination)
                raster_tiles.append(path)
                console.print(f"  LiDAR tile {index}: {path.name}")
            except Exception as exc:
                console.print(f"[yellow]  LiDAR tile {index} failed:[/yellow] {exc}")
    else:
        raise typer.BadParameter("Provide --dtm-dir or use --download-lidar with a bbox")
    if not raster_tiles:
        raise typer.BadParameter("No DTM GeoTIFF tiles were available")

    from rasterio import open as rio_open
    features_by_tile: dict[Path, list[dict]] = {}
    for raster in raster_tiles:
        with rio_open(raster) as dataset:
            bounds = (dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top)
        features_by_tile[raster] = query_features_for_tile(bounds, db_url)
    stamped = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = project / "training" / "datasets" / f"dataset-{stamped}"
    dataset_path, counts = build_dataset(
        raster_tiles,
        features_by_tile,
        output_dir,
        negative_multiplier=negative_multiplier,
        max_features=max_features,
    )
    manifest_rows = []
    manifest_path = output_dir / "examples.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            manifest_rows.append(__import__("json").loads(line))
    registered = register_training_manifest(manifest_rows, db_url)
    console.print(f"[green]Training dataset created:[/green] {dataset_path}")
    console.print(f"  Examples: {counts['examples']} (positive {counts['positive']}, background {counts['negative']})")
    console.print(f"  Database examples recorded: {registered}")


def query_features_for_tile(bounds: tuple[float, float, float, float], database_url: str) -> list[dict]:
    from prospector.training.db import query_features
    return query_features(bounds, database_url=database_url)


@training_app.command("fit")
def train_fit(
    dataset: Path = typer.Option(..., "--dataset", help="training-dataset.npz produced by build-dataset."),
    project: Path = typer.Option(Path("./project"), "--project", "-p"),
    database_url: str | None = typer.Option(None, "--database-url", envvar="PROSPECTOR_DATABASE_URL", show_default=False),
) -> None:
    """Fit the persisted local archaeology/background classifier."""
    result = train_model(dataset, project / "models")
    register_model(
        model_name="prospector-archaeology",
        version=result.version,
        model_path=str(result.model_path),
        dataset_path=str(dataset),
        metrics=result.metrics,
        feature_schema={"version": "patch-v1", "dimensions": 3 * 32 * 32 + 16 + 8},
        database_url=_database_url(database_url),
    )
    console.print(f"[green]Model trained:[/green] {result.version}")
    console.print(f"  Model: {result.model_path}")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if key in result.metrics:
            console.print(f"  {key}: {result.metrics[key]:.3f}")


@training_app.command("status")
def train_status(
    database_url: str | None = typer.Option(None, "--database-url", envvar="PROSPECTOR_DATABASE_URL", show_default=False),
) -> None:
    """Show training corpus and registered model counts."""
    db_url = _database_url(database_url)
    ensure_schema(db_url)
    with connect(db_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT source_layer, count(*) FROM aim_features GROUP BY source_layer ORDER BY source_layer")
            features = cursor.fetchall()
            cursor.execute("SELECT count(*) FROM training_examples")
            example_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT model_name, version, created_at FROM model_registry ORDER BY created_at DESC LIMIT 5")
            models = cursor.fetchall()
    console.print("[bold]Training corpus[/bold]")
    for layer, count in features:
        console.print(f"  {layer}: {count}")
    console.print(f"  training examples recorded: {example_count}")
    if models:
        console.print("[bold]Recent models[/bold]")
        for name, version, created_at in models:
            console.print(f"  {name} {version} ({created_at})")
