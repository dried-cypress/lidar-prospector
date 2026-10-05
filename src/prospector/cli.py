from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from prospector import __version__
from prospector.config import AppConfig
from prospector.domain import Coordinate, StudyArea
from prospector.providers.historic_england import HistoricEnglandProvider
from prospector.providers.imagery import HighResolutionImageryProvider
from prospector.providers.http import CachedResponse, HttpClient
from prospector.providers.lidar import DTM_COVERAGE_ID, WCS_URL, LidarProvider
from prospector.providers.location import DEFAULT_REVERSE_GEOCODER_URL, LocationProvider
from prospector.providers.openstreetmap import OpenStreetMapContextProvider
from prospector.providers.os import OSFeaturesProvider, OSOpenMapProvider
from prospector.providers.satellite import SatelliteProvider
from prospector.reporting.geojson import write_candidate_collection, write_feature_collection
from prospector.reporting.html import write_html_report
from prospector.reporting.overlay import (
    create_lidar_aim_overlay,
    create_lidar_anomaly_aim_overlay,
    create_lidar_anomaly_overlay,
    create_lidar_base_overlay,
)
from prospector.runs import create_run, sha256_file, update_run_metadata
from prospector.terrain.anomalies import detect_terrain_anomalies
from prospector.terrain.context import build_modern_context_raster
from prospector.terrain.derivatives import hillshade

app = typer.Typer(
    name="prospector",
    help="Archaeological landscape prospection.",
    no_args_is_help=True,
)
console = Console()


def _cache_manifest(client: HttpClient, entries: list[CachedResponse]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for entry in entries:
        metadata_path = entry.metadata_path
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        try:
            relative_cache = str(entry.path.relative_to(client.cache_dir))
        except ValueError:
            relative_cache = str(entry.path)
        manifest.append(
            {
                "cache_hit": entry.cache_hit,
                "sha256": entry.sha256,
                "size_bytes": metadata.get("size_bytes"),
                "content_type": metadata.get("content_type"),
                "url": metadata.get("url"),
                "params": metadata.get("params"),
                "cache_path": relative_cache,
            }
        )
    return manifest


def _relative_output(path: Path | None, run_path: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(run_path))
    except ValueError:
        return str(path)


def _raster_bounds(path: Path) -> tuple[float, float, float, float]:
    import rasterio

    with rasterio.open(path) as dataset:
        return (
            float(dataset.bounds.left),
            float(dataset.bounds.bottom),
            float(dataset.bounds.right),
            float(dataset.bounds.top),
        )


def _require_project_venv() -> None:
    """Prevent an old user-site console script from running the application.

    Prospector's supported installation is the repository-local `.venv`. A
    user-site installation can otherwise win PATH resolution and mix system and
    user Python packages, which is particularly dangerous for compiled geo
    dependencies such as Rasterio and Matplotlib.
    """
    virtual_env = os.getenv("VIRTUAL_ENV")
    if not virtual_env:
        raise RuntimeError(
            "Prospector must run from the project virtual environment. "
            "Run `source .venv/bin/activate` first, then run `prospector`, "
            "or invoke `.venv/bin/prospector` directly."
        )

    active = Path(sys.prefix).resolve()
    expected = Path(virtual_env).resolve()
    if active != expected:
        raise RuntimeError(
            f"Prospector is running under {sys.executable}, not the active virtual "
            f"environment at {expected}. Use `{expected / 'bin/prospector'}` instead."
        )

    project_venv = Path(__file__).resolve().parents[2] / ".venv"
    if project_venv.exists() and active != project_venv.resolve():
        raise RuntimeError(
            "Prospector is not running from this project's .venv. "
            f"Use `{project_venv / 'bin/prospector'}` or activate that environment."
        )


@app.command()
def version() -> None:
    """Show the application version."""
    console.print(__version__)


@app.command()
def init(
    project: Path = typer.Option(Path("./project"), "--project", "-p"),
) -> None:
    """Initialise a project directory."""
    config = AppConfig.for_project(project)
    config.ensure_directories()
    console.print(f"[green]Project initialised:[/green] {project.resolve()}")


@app.command()
def analyse(
    latitude: float = typer.Option(..., "--latitude", help="WGS84 latitude."),
    longitude: float = typer.Option(..., "--longitude", help="WGS84 longitude."),
    diameter: float | None = typer.Option(
        None,
        "--diameter",
        help="Study-area diameter in metres. Defaults to the project configuration.",
    ),
    project: Path = typer.Option(Path("./project"), "--project", "-p"),
    no_lidar_download: bool = typer.Option(
        False,
        "--no-lidar-download",
        help="Discover LiDAR but do not request the raster.",
    ),
    no_context: bool = typer.Option(
        False,
        "--no-context",
        help="Disable all external modern/satellite context acquisition.",
    ),
    no_os: bool = typer.Option(
        False,
        "--no-os",
        help="Disable automatic OS OpenMap Local acquisition.",
    ),
    no_osm: bool = typer.Option(
        False,
        "--no-osm",
        help="Disable supplementary OpenStreetMap modern-feature acquisition.",
    ),
    no_satellite: bool = typer.Option(
        False,
        "--no-satellite",
        help="Disable Sentinel-2 contextual imagery acquisition.",
    ),
    no_location: bool = typer.Option(
        False,
        "--no-location",
        help="Disable reverse-geocoding of the study coordinate to a place name.",
    ),
    os_data: Path | None = typer.Option(
        None,
        "--os-data",
        help="Use a local OS OpenMap Local ZIP, GeoPackage, shapefile or directory instead of downloading it.",
    ),
    os_api_key: str | None = typer.Option(
        None,
        "--os-api-key",
        envvar="OS_DATAHUB_API_KEY",
        help="OS Data Hub API key for fast AOI-scoped OS Features API context. OS_API_KEY remains supported as a legacy fallback.",
        show_default=False,
    ),
    os_download: bool = typer.Option(
        False,
        "--os-download",
        help="Allow the slower OS OpenMap Local 100 km National Grid tile download when no OS API key is available.",
    ),
    sensitivity: int = typer.Option(
        5,
        "--sensitivity",
        min=1,
        max=10,
        help="Anomaly detection sensitivity from 1 (conservative) to 10 (maximum exploratory recall).",
    ),
    workers: int = typer.Option(
        0,
        "--workers",
        min=0,
        help="Parallel detector workers. 0 selects an automatic safe default.",
    ),
) -> None:
    """Analyse a circular study area centred on a WGS84 coordinate."""
    _require_project_venv()
    os_api_key = os_api_key or os.getenv("OS_API_KEY")
    config = AppConfig.for_project(project)
    config.ensure_directories()
    requested_diameter = config.default_diameter_m if diameter is None else diameter
    try:
        area = StudyArea.from_coordinate(Coordinate(latitude, longitude), requested_diameter)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    run = create_run(config, area)
    client = HttpClient(config.cache_dir, config.request_timeout_seconds)
    run_path = Path(run.path)
    errors: list[str] = []
    location_name: str | None = None
    location_display_name: str | None = None
    location_attribution: str | None = None
    location_cache: list[CachedResponse] = []
    location_metadata: dict[str, Any] = {"enabled": False}
    if not no_location:
        location_url = os.getenv("PROSPECTOR_GEOCODER_URL", DEFAULT_REVERSE_GEOCODER_URL)
        location_result = LocationProvider(client, location_url).reverse(latitude, longitude)
        location_name = location_result.name
        location_display_name = location_result.display_name
        location_attribution = (location_result.metadata or {}).get("attribution")
        location_metadata = {"enabled": True, **location_result.metadata}
        if location_result.cache_entry is not None:
            location_cache.append(location_result.cache_entry)
        if location_result.error:
            errors.append(f"Location lookup failed: {location_result.error}")
        elif location_name:
            console.print(f"Location: {location_name}")
    else:
        console.print("Location lookup disabled")
    study_bbox = area.bounds
    if not (
        study_bbox[0] <= area.easting <= study_bbox[2]
        and study_bbox[1] <= area.northing <= study_bbox[3]
    ):
        raise RuntimeError("Study-area centre is outside its own EPSG:27700 bounding box")

    console.print(f"[bold]Run:[/bold] {run.run_id}")
    console.print(f"Centre: {latitude:.7f}, {longitude:.7f}")
    console.print(f"Diameter: {requested_diameter:.1f} m")
    console.print(f"British National Grid: {area.easting:.2f} E, {area.northing:.2f} N")
    console.print(
        "Study bbox: "
        f"{study_bbox[0]:.2f}, {study_bbox[1]:.2f}, "
        f"{study_bbox[2]:.2f}, {study_bbox[3]:.2f} (EPSG:27700)"
    )

    # Historic England AIM acquisition. This successful V0.2.1 source and all
    # existing LiDAR overlay rendering are intentionally unchanged.
    console.print("[bold]Historic England Aerial Archaeology Mapping[/bold]")
    he = HistoricEnglandProvider(client)
    aim_errors: list[str] = []
    he_metadata: dict[str, Any] = {}
    try:
        aim_result = he.search(area.geometry_wkt)
        aim = aim_result.features
        monument_extents = aim_result.monument_extents
        project_areas = aim_result.project_areas
        he_cache = aim_result.cache_entries
        aim_errors = aim_result.errors
        he_metadata = aim_result.metadata
        errors.extend(aim_errors)
    except Exception as exc:
        aim = []
        monument_extents = []
        project_areas = []
        he_cache = []
        he_metadata = {"status": "failed"}
        errors.append(f"Historic England acquisition failed: {exc}")
    aim_output = run_path / "sources" / "historic-england-detailed-mapping.geojson"
    monument_extents_output = run_path / "sources" / "historic-england-monument-extents.geojson"
    project_areas_output = run_path / "sources" / "historic-england-project-areas.geojson"
    write_feature_collection(aim, aim_output)
    write_feature_collection(monument_extents, monument_extents_output)
    write_feature_collection(project_areas, project_areas_output)
    he_hits = sum(1 for item in he_cache if item.cache_hit)
    console.print(f"  Detailed mapping features: {len(aim)}")
    console.print(f"  Monument extents: {len(monument_extents)}")
    console.print(f"  Project areas: {len(project_areas)}")
    for layer_name, layer_info in he_metadata.get("layers", {}).items():
        console.print(
            f"    {layer_name}: {layer_info.get('records_returned', 0)} record(s), "
            f"{layer_info.get('pages_returned', 0)} page(s)"
        )
    if he_metadata.get("truncated"):
        console.print(
            "[yellow]  WARNING: Historic England returned a truncated layer; "
            "results are incomplete.[/yellow]"
        )
    raster_project_count = int(he_metadata.get("raster_only_project_count", 0))
    if raster_project_count:
        console.print(
            f"[yellow]  NOTE: {raster_project_count} Historic England project area(s) "
            "in the AOI are raster-only; their detailed hand-drawn features are "
            "not available through the downloadable vector FeatureServer.[/yellow]"
        )
    console.print(f"  Cache: {he_hits} hit(s), {len(he_cache) - he_hits} download(s)")

    # LiDAR acquisition and the locked V0.2.1 renderer.
    console.print("[bold]NLS Maps LiDAR DTM 50cm-1m — England component[/bold]")
    lidar = LidarProvider(client)
    coverage_id: str | None = None
    capabilities_cache: CachedResponse | None = None
    dtm_output: Path | None = None
    hillshade_output: Path | None = None
    lidar_cache: CachedResponse | None = None
    try:
        capabilities, capabilities_cache = lidar.capabilities()
        coverage_id = lidar.coverage_id(capabilities)
        capabilities_output = run_path / "sources" / "lidar-wcs-capabilities.xml"
        capabilities_output.write_bytes(capabilities_cache.path.read_bytes())
        console.print(f"  Coverage: {coverage_id}")
        console.print("  WCS capabilities: " + ("cache hit" if capabilities_cache.cache_hit else "downloaded"))
    except Exception as exc:
        errors.append(f"LiDAR capabilities failed: {exc}")
        console.print(f"[yellow]  Capabilities failed:[/yellow] {exc}")

    if coverage_id is not None and not no_lidar_download:
        try:
            dtm_output = run_path / "terrain" / "dtm.tif"
            dtm_output, lidar_cache = lidar.download_geotiff(study_bbox, dtm_output)
            console.print("[green]  DTM:[/green] " + ("cache hit" if lidar_cache.cache_hit else "downloaded"))
            hillshade_output = run_path / "terrain" / "hillshade.tif"
            hillshade(dtm_output, hillshade_output)
            console.print(f"[green]  Hillshade generated:[/green] {hillshade_output}")
        except Exception as exc:
            errors.append(f"LiDAR download/processing failed: {exc}")
            console.print(f"[yellow]  LiDAR processing failed:[/yellow] {exc}")
    elif no_lidar_download:
        console.print("  DTM download skipped by --no-lidar-download")

    # External contextual evidence is acquired only after the DTM succeeds so
    # every contextual raster can be snapped to the same exact LiDAR grid.
    modern_features: list[dict[str, Any]] = []
    context_cache: list[CachedResponse] = []
    context_metadata: dict[str, Any] = {"enabled": not no_context}
    modern_context_geojson = run_path / "sources" / "modern-context.geojson"
    satellite_support_path: Path | None = None
    satellite_preview_path: Path | None = None
    satellite_search_path: Path | None = None
    satellite_metadata: dict[str, Any] = {"enabled": False}
    highres_imagery_path: Path | None = None
    highres_imagery_metadata: dict[str, Any] = {"enabled": False}

    if dtm_output is not None and dtm_output.is_file() and not no_context:
        console.print("[bold]Modern feature context — Ordnance Survey[/bold]")
        if no_os and os_data is not None:
            errors.append("--os-data cannot be used together with --no-os; local OS data was ignored")
        elif not no_os:
            os_start = time.perf_counter()
            bbox = _raster_bounds(dtm_output)
            try:
                # The fast path asks the OS Features API for only the features
                # intersecting this AOI. It avoids the very large OpenMap Local
                # 100 km National Grid download used by earlier versions.
                if os_api_key and os_data is None:
                    os_result = OSFeaturesProvider(client, os_api_key).search(bbox)
                    context_metadata["os_features_api"] = os_result.metadata
                    console.print(f"  OS Features API features: {len(os_result.features)}")
                elif os_data is not None:
                    os_result = OSOpenMapProvider(client).search(
                        bbox, run_path / "sources" / "os-openmap", os_data
                    )
                    context_metadata["os_openmap"] = os_result.metadata
                    console.print(f"  Local OS OpenMap features: {len(os_result.features)}")
                elif os_download:
                    os_result = OSOpenMapProvider(client).search(
                        bbox, run_path / "sources" / "os-openmap", None
                    )
                    context_metadata["os_openmap"] = os_result.metadata
                    console.print(f"  OS OpenMap features: {len(os_result.features)}")
                else:
                    os_result = None
                    context_metadata["os_openmap"] = {
                        "enabled": False,
                        "reason": "no OS_API_KEY configured; slow 100 km OpenMap Local download disabled by default",
                    }
                    console.print(
                        "  OS OpenMap Local download skipped (no OS_API_KEY). "
                        "Using OSM supplement; use --os-api-key or --os-download to enable OS data."
                    )
                if os_result is not None:
                    modern_features.extend(os_result.features)
                    context_cache.extend(os_result.cache_entries)
                    errors.extend(os_result.errors)
            except Exception as exc:
                errors.append(f"OS context processing failed: {exc}")
                console.print(f"[yellow]  OS context failed:[/yellow] {exc}")
            finally:
                console.print(f"  OS context elapsed: {time.perf_counter() - os_start:.1f}s")
        else:
            context_metadata["os_openmap"] = {"enabled": False}
            console.print("  OS acquisition disabled")

        console.print("[bold]Modern feature context — OpenStreetMap supplement[/bold]")
        if not no_osm:
            try:
                osm_result = OpenStreetMapContextProvider(client).search(_raster_bounds(dtm_output))
                modern_features.extend(osm_result.features)
                context_cache.extend(osm_result.cache_entries)
                errors.extend(osm_result.errors)
                context_metadata["openstreetmap"] = osm_result.metadata
                console.print(f"  OSM context features: {len(osm_result.features)}")
            except Exception as exc:
                errors.append(f"OpenStreetMap context processing failed: {exc}")
                console.print(f"[yellow]  OSM context failed:[/yellow] {exc}")
        else:
            context_metadata["openstreetmap"] = {"enabled": False}
            console.print("  OSM acquisition disabled")

        # Deduplicate identical geometries returned by overlapping providers.
        deduped: dict[tuple[str, bytes], dict[str, Any]] = {}
        for feature in modern_features:
            geometry = feature.get("geometry")
            properties = feature.get("properties") or {}
            if not geometry:
                continue
            try:
                from shapely.geometry import shape

                geometry_obj = shape(geometry)
                key = (str(properties.get("prospector_context_type", "")), geometry_obj.wkb)
            except Exception:
                continue
            deduped[key] = feature
        modern_features = list(deduped.values())
        write_feature_collection(modern_features, modern_context_geojson)
        console.print(f"  Combined modern-context features: {len(modern_features)}")

        if modern_features:
            modern_raster = build_modern_context_raster(
                dtm_output,
                modern_features,
                run_path / "terrain" / "modern-context-score.tif",
            )
            modern_context_path = modern_raster.score_path
            context_metadata["raster"] = modern_raster.metadata
        else:
            modern_context_path = None
    else:
        write_feature_collection([], modern_context_geojson)
        modern_context_path = None
        if no_context:
            console.print("[bold]External contextual evidence disabled[/bold]")
        else:
            console.print("[yellow]External context skipped because no DTM is available[/yellow]")

        if no_context:
            context_metadata = {"enabled": False}

    if dtm_output is not None and dtm_output.is_file() and not no_context and not no_satellite:
        console.print("[bold]Satellite context — Sentinel-2 L2A[/bold]")
        try:
            satellite_result = SatelliteProvider(client).acquire(
                _raster_bounds(dtm_output), dtm_output, run_path
            )
            satellite_support_path = satellite_result.support_path
            satellite_preview_path = satellite_result.preview_path
            satellite_search_path = satellite_result.search_path
            satellite_metadata = satellite_result.metadata
            context_cache.extend(satellite_result.cache_entries)
            errors.extend(satellite_result.errors)
            console.print(f"  Selected scenes: {len(satellite_result.scenes)}")
            if satellite_support_path:
                console.print(f"[green]  Satellite support raster:[/green] {satellite_support_path}")
            if satellite_preview_path:
                # v0.4.3 no longer uses Sentinel-2 imagery as a visual base layer.
                satellite_preview_path.unlink(missing_ok=True)
                satellite_preview_path = None
        except Exception as exc:
            errors.append(f"Satellite context processing failed: {exc}")
            console.print(f"[yellow]  Satellite context failed:[/yellow] {exc}")
    elif not no_satellite and dtm_output is None:
        satellite_metadata = {"enabled": False, "reason": "no DTM"}

    if dtm_output is not None and dtm_output.is_file() and not no_satellite:
        console.print("[bold]High-resolution imagery — Esri World Imagery[/bold]")
        try:
            imagery_result = HighResolutionImageryProvider(client).acquire(
                _raster_bounds(dtm_output), run_path
            )
            highres_imagery_path = imagery_result.preview_path
            highres_imagery_metadata = imagery_result.metadata
            context_cache.extend(imagery_result.cache_entries)
            errors.extend(imagery_result.errors)
            if highres_imagery_path:
                console.print(f"[green]  Imagery:[/green] {highres_imagery_path}")
        except Exception as exc:
            errors.append(f"High-resolution imagery processing failed: {exc}")
            console.print(f"[yellow]  High-resolution imagery failed:[/yellow] {exc}")
    elif not no_satellite and dtm_output is None:
        highres_imagery_metadata = {"enabled": False, "reason": "no DTM"}

    overlay_output: Path | None = None
    anomaly_overlay_output: Path | None = None
    anomaly_aim_overlay_output: Path | None = None
    lidar_base_output: Path | None = None
    anomaly_base_output: Path | None = None
    candidate_output: Path | None = None
    candidates = []
    anomaly_metadata: dict[str, Any] = {}
    map_bounds = study_bbox
    if dtm_output is not None and dtm_output.is_file():
        try:
            map_bounds = _raster_bounds(dtm_output)
            lidar_base_output = run_path / "overlays" / "lidar-base.png"
            anomaly_base_output = run_path / "overlays" / "lidar-anomalies-base.png"
            create_lidar_base_overlay(dtm_output, lidar_base_output, anomaly_base=False)
            create_lidar_base_overlay(dtm_output, anomaly_base_output, anomaly_base=True)
            overlay_output = run_path / "overlays" / "lidar-aim.png"
            create_lidar_aim_overlay(
                dtm_output,
                aim,
                area.geometry_wkt,
                area.easting,
                area.northing,
                overlay_output,
                hillshade_path=hillshade_output,
                render_bounds=map_bounds,
                monument_extents=monument_extents,
                project_areas=project_areas,
            )
            console.print(f"[green]HE-styled LiDAR overlay generated:[/green] {overlay_output}")
        except Exception as exc:
            errors.append(f"Overlay generation failed: {exc}")
            console.print(f"[yellow]Overlay generation failed:[/yellow] {exc}")

        try:
            console.print("[bold]Context-aware LiDAR terrain anomaly scan[/bold]")
            console.print(f"  Detection sensitivity: {sensitivity}")
            console.print(f"  Detector workers: {workers if workers > 0 else 'auto'}")
            candidates, anomaly_metadata = detect_terrain_anomalies(
                dtm_output,
                aim,
                monument_extents=monument_extents,
                sensitivity=sensitivity,
                workers=workers,
                modern_features=modern_features,
                modern_context_path=modern_context_path,
                satellite_support_path=satellite_support_path,
            )
            anomaly_metadata["external_context"] = {
                **context_metadata,
                "satellite": satellite_metadata,
                "high_resolution_imagery": highres_imagery_metadata,
                "location": location_metadata,
                "modern_feature_count": len(modern_features),
            }
            candidate_output = run_path / "candidates" / "terrain-anomalies.geojson"
            write_candidate_collection(candidates, candidate_output)
            anomaly_overlay_output = run_path / "overlays" / "lidar-anomalies.png"
            create_lidar_anomaly_overlay(
                dtm_output, candidates, anomaly_overlay_output, hillshade_path=hillshade_output
            )
            anomaly_aim_overlay_output = run_path / "overlays" / "lidar-anomalies-aim.png"
            create_lidar_anomaly_aim_overlay(
                dtm_output, candidates, aim, anomaly_aim_overlay_output,
                monument_extents=monument_extents,
                project_areas=project_areas,
            )
            he_reference = anomaly_metadata.get("historic_england_reference", {})
            if he_reference.get("known_validation_total", 0):
                console.print(
                    "  HE validation: "
                    f"{he_reference.get('known_validation_validated_hits', 0)}/"
                    f"{he_reference.get('known_validation_total', 0)} known features validated "
                    f"({he_reference.get('known_validation_recall_percent', 0.0):.1f}%)"
                )
                validation_output = run_path / "candidates" / "historic-england-detection-validation.geojson"
                validation_features = []
                for item in anomaly_metadata.get("historic_england_validation", []):
                    source_index = int(item.get("source_index", -1))
                    if not 0 <= source_index < len(aim):
                        continue
                    validation_features.append({
                        "type": "Feature",
                        "geometry": aim[source_index].get("geometry"),
                        "properties": item,
                    })
                write_feature_collection(validation_features, validation_output)
                anomaly_metadata["historic_england_validation_geojson"] = str(validation_output)
            console.print(f"  Ranked candidates retained: {len(candidates)}")
            console.print(f"[green]Anomaly overlay generated:[/green] {anomaly_overlay_output}")
            console.print(f"[green]Combined anomaly + AIM overlay generated:[/green] {anomaly_aim_overlay_output}")
        except Exception as exc:
            errors.append(f"LiDAR anomaly analysis failed: {exc}")
            console.print(f"[yellow]LiDAR anomaly analysis failed:[/yellow] {exc}")

    report_output = run_path / "reports" / "report.html"
    cache_entries = he_cache[:]
    if capabilities_cache is not None:
        cache_entries.append(capabilities_cache)
    if lidar_cache is not None:
        cache_entries.append(lidar_cache)
    cache_entries.extend(context_cache)
    cache_entries.extend(location_cache)
    cache_manifest = _cache_manifest(client, cache_entries)
    try:
        write_html_report(
            report_output,
            run_id=run.run_id,
            application_version=__version__,
            latitude=latitude,
            longitude=longitude,
            easting=area.easting,
            northing=area.northing,
            diameter_m=requested_diameter,
            aim_features=aim,
            monument_extents=monument_extents,
            project_areas=project_areas,
            cache_entries=cache_manifest,
            errors=errors,
            overlay_path=overlay_output,
            dtm_path=dtm_output,
            hillshade_path=hillshade_output,
            aim_geojson_path=aim_output,
            anomaly_overlay_path=anomaly_overlay_output,
            anomaly_aim_overlay_path=anomaly_aim_overlay_output,
            candidates=candidates,
            candidate_geojson_path=candidate_output,
            anomaly_metadata=anomaly_metadata,
            location_name=location_name,
            location_display_name=location_display_name,
            location_attribution=location_attribution,
            study_bounds=study_bbox,
            map_bounds=map_bounds,
            modern_context_path=modern_context_geojson,
            modern_context_raster_path=(run_path / "terrain" / "modern-context-score.tif") if (run_path / "terrain" / "modern-context-score.tif").is_file() else None,
            satellite_preview_path=None,
            high_resolution_imagery_path=highres_imagery_path,
            satellite_support_path=satellite_support_path,
            satellite_search_path=satellite_search_path,
            lidar_base_path=lidar_base_output,
            anomaly_base_path=anomaly_base_output,
        )
        console.print(f"[green]HTML report generated:[/green] {report_output}")
    except Exception as exc:
        errors.append(f"HTML report generation failed: {exc}")
        console.print(f"[yellow]HTML report generation failed:[/yellow] {exc}")

    outputs: dict[str, object] = {
        "historic_england_detailed_mapping_geojson": _relative_output(aim_output, run_path),
        "historic_england_monument_extents_geojson": _relative_output(monument_extents_output, run_path),
        "historic_england_project_areas_geojson": _relative_output(project_areas_output, run_path),
        "lidar_base": _relative_output(lidar_base_output, run_path),
        "lidar_anomalies_base": _relative_output(anomaly_base_output, run_path),
        "modern_context_geojson": _relative_output(modern_context_geojson, run_path),
        "modern_context_raster": _relative_output((run_path / "terrain" / "modern-context-score.tif") if (run_path / "terrain" / "modern-context-score.tif").is_file() else None, run_path),
        "satellite_support": _relative_output(satellite_support_path, run_path),
        "high_resolution_imagery": _relative_output(highres_imagery_path, run_path),
        "satellite_search": _relative_output(satellite_search_path, run_path),
        "dtm": _relative_output(dtm_output, run_path),
        "hillshade": _relative_output(hillshade_output, run_path),
        "overlay": _relative_output(overlay_output, run_path),
        "anomaly_overlay": _relative_output(anomaly_overlay_output, run_path),
        "anomaly_aim_overlay": _relative_output(anomaly_aim_overlay_output, run_path),
        "candidate_geojson": _relative_output(candidate_output, run_path),
        "historic_england_detection_validation": _relative_output((run_path / "candidates" / "historic-england-detection-validation.geojson") if (run_path / "candidates" / "historic-england-detection-validation.geojson").is_file() else None, run_path),
        "html_report": _relative_output(report_output, run_path) if report_output.is_file() else None,
    }
    checksum_paths = {
        "historic_england_detailed_mapping_geojson": aim_output,
        "historic_england_monument_extents_geojson": monument_extents_output,
        "historic_england_project_areas_geojson": project_areas_output,
        "modern_context_geojson": modern_context_geojson,
        "modern_context_raster": (run_path / "terrain" / "modern-context-score.tif"),
        "satellite_support": satellite_support_path,
        "high_resolution_imagery": highres_imagery_path,
        "satellite_search": satellite_search_path,
        "dtm": dtm_output,
        "hillshade": hillshade_output,
        "overlay": overlay_output,
        "anomaly_overlay": anomaly_overlay_output,
        "anomaly_aim_overlay": anomaly_aim_overlay_output,
        "candidate_geojson": candidate_output,
        "html_report": report_output if report_output.is_file() else None,
    }
    checksums = {key: sha256_file(path) for key, path in checksum_paths.items() if path is not None and path.is_file()}

    update_run_metadata(
        run,
        status="completed" if not errors else "completed_with_warnings",
        analysis={
            "detector": {
                **anomaly_metadata,
                "candidate_count": len(candidates),
                "status": "completed" if candidate_output is not None else "failed",
            },
            "classifier": {
                "name": "HE-guided terrain-signature ranker",
                "version": __version__,
                "status": "deterministic morphology + unsupervised per-AOI ML novelty + HE reference signatures",
            },
        },
        sources={
            "historic_england": {
                "provider": "Historic England Aerial Investigation and Mapping",
                "layers": {
                    "Detailed_Mapping": len(aim),
                    "Monument_Extents": len(monument_extents),
                    "Project_Area": len(project_areas),
                },
                "feature_count": len(aim) + len(monument_extents) + len(project_areas),
                "cache_entries": cache_manifest[: len(he_cache)],
                "query": he_metadata,
                "errors": aim_errors,
            },
            "nls_lidar": {
                "provider": "NLS Maps LiDAR DTM 50cm-1m composite (England component)",
                "service_provider": "Environment Agency",
                "service_type": "WCS 2.0.1",
                "wcs_url": WCS_URL,
                "coverage_id": coverage_id,
                "expected_coverage_id": DTM_COVERAGE_ID,
                "cache_entries": cache_manifest[len(he_cache) : len(he_cache) + (2 if capabilities_cache is not None else 0) + (1 if lidar_cache is not None else 0)],
            },
            "context": {
                **context_metadata,
                "satellite": satellite_metadata,
                "location": location_metadata,
                "modern_feature_count": len(modern_features),
            },
        },
        outputs=outputs | {"sha256": checksums},
        errors=errors,
        finished=True,
    )
    console.print(f"[bold]Outputs:[/bold] {run_path.resolve()}")
    if errors:
        console.print(f"[yellow]Warnings:[/yellow] {len(errors)}")
        for warning in errors:
            console.print(f"[yellow]  • {warning}[/yellow]")


if __name__ == "__main__":
    app()
