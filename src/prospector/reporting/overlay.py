from __future__ import annotations

from pathlib import Path
from typing import Any

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
    shape,
)
from shapely import wkt

# Historic England Aerial Investigation & Mapping Explorer symbology.
# HE's live Detailed_Mapping service currently stores these values in uppercase
# (e.g. BANK, DITCH and SCARP_SLOPE_EDGE), so the resolver below deliberately
# accepts both the published labels and the service values.
HE_LAYER_STYLES: dict[str, dict[str, Any]] = {
    "Bank": {
        "colour": "#A50026",
        "line_alpha": 0.96,
        "fill_alpha": 0.10,
        "dash": None,
        "description": "Banks, platforms, mounds and spoil heaps",
    },
    "Ditch": {
        "colour": "#313695",
        "line_alpha": 0.97,
        "fill_alpha": 0.08,
        "dash": None,
        "description": "Cut features such as ditches, ponds, pits and hollow ways",
    },
    "Extent of Feature": {
        "colour": "#FDAE61",
        "line_alpha": 0.94,
        "fill_alpha": 0.07,
        "dash": (7, 4),
        "description": "Large area extents such as airfields, military camps or extraction",
    },
    "Ridge and Furrow Alignment": {
        "colour": "#74ADD1",
        "line_alpha": 0.96,
        "fill_alpha": 0.04,
        "dash": None,
        "description": "Ridge-and-furrow orientation",
    },
    "Ridge and Furrow Area": {
        "colour": "#74ADD1",
        "line_alpha": 0.96,
        "fill_alpha": 0.06,
        "dash": (2, 4),
        "description": "Blocks of ridge and furrow",
    },
    "Slope": {
        "colour": "#4575B4",
        "line_alpha": 0.96,
        "fill_alpha": 0.06,
        "dash": None,
        "description": "Scarps, platform edges and other slope features",
    },
    "Structure": {
        "colour": "#F46D43",
        "line_alpha": 0.95,
        "fill_alpha": 0.09,
        "dash": None,
        "description": "Structures including buildings and other built features",
    },
}

# Neutral rather than purple: unknown/new source symbology should be obvious
# without pretending that it belongs to a Historic England layer.
HE_CONTEXT_STYLES: dict[str, dict[str, Any]] = {
    "Monument_Extents": {
        "colour": "#666666",
        "line_alpha": 0.92,
        "fill_alpha": 0.035,
        "dash": None,
        "description": "General extent of archaeological monuments",
    },
    "Project_Area": {
        "colour": "#D73027",
        "line_alpha": 0.90,
        "fill_alpha": 0.025,
        "dash": (8, 5),
        "description": "Extent of a Historic England Aerial Investigation and Mapping project",
    },
}

DEFAULT_HE_STYLE = {
    "colour": "#5B6470",
    "line_alpha": 0.70,
    "fill_alpha": 0.04,
    "dash": (3, 3),
    "description": "Unmapped Historic England layer",
}

_LAYER_ALIASES: dict[str, str] = {
    "bank": "Bank",
    "banks": "Bank",
    "ditch": "Ditch",
    "ditches": "Ditch",
    "extent of feature": "Extent of Feature",
    "extent_of_feature": "Extent of Feature",
    "ridge and furrow alignment": "Ridge and Furrow Alignment",
    "ridge_and_furrow_alignment": "Ridge and Furrow Alignment",
    "ridge and furrow area": "Ridge and Furrow Area",
    "ridge_and_furrow_area": "Ridge and Furrow Area",
    "slope": "Slope",
    "scarp slope edge": "Slope",
    "scarp_slope_edge": "Slope",
    "scarp slope/edge": "Slope",
    "structure": "Structure",
    "structures": "Structure",
}


def require_plotting_stack() -> None:
    try:
        import matplotlib  # noqa: F401
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Overlay generation requires the geospatial extras. "
            "Install with: pip install -e '.[geo]'"
        ) from exc


def _normalise_layer(value: object) -> str:
    text = " ".join(str(value or "").strip().casefold().replace("_", " ").split())
    return text


def he_style(layer: str | None, monument_type: str | None = None, evidence: str | None = None) -> dict[str, Any]:
    """Resolve Historic England detailed-mapping symbology robustly.

    Historic England's current live service uses uppercase type IDs. Unknown
    values are kept neutral rather than silently becoming the old purple default.
    """
    normalised = _normalise_layer(layer)
    if normalised in {"monument extents", "monument_extent", "monument_extents"}:
        return HE_CONTEXT_STYLES["Monument_Extents"]
    if normalised in {"project area", "project_area", "project areas", "project_areas"}:
        return HE_CONTEXT_STYLES["Project_Area"]
    if normalised in _LAYER_ALIASES:
        return HE_LAYER_STYLES[_LAYER_ALIASES[normalised]]

    # Defensive fallbacks for older exports with missing LAYER values. The
    # fallback remains based on physical form, matching HE's published approach.
    semantic = _normalise_layer(f"{monument_type or ''} {evidence or ''}")
    if any(token in semantic for token in ("ditch", "pit", "hollow", "pond", "cut")):
        return HE_LAYER_STYLES["Ditch"]
    if any(token in semantic for token in ("bank", "mound", "barrow", "embankment", "platform", "earthwork")):
        return HE_LAYER_STYLES["Bank"]
    if "ridge" in semantic and "furrow" in semantic:
        return HE_LAYER_STYLES["Ridge and Furrow Area"]
    if any(token in semantic for token in ("slope", "scarp")):
        return HE_LAYER_STYLES["Slope"]
    if any(token in semantic for token in ("structure", "building", "hut", "mast", "camp")):
        return HE_LAYER_STYLES["Structure"]
    return DEFAULT_HE_STYLE


def _plot_geometry(
    ax: Any,
    geometry: Any,
    *,
    style: dict[str, Any],
    linewidth: float = 1.2,
    halo: bool = True,
) -> None:
    """Plot a geometry with a high-contrast outline so thin AIM features survive rasterisation."""
    if geometry.is_empty:
        return
    colour = style["colour"]
    line_alpha = float(style.get("line_alpha", 1.0))
    fill_alpha = float(style.get("fill_alpha", 0.0))
    dash = style.get("dash")
    linestyle = (0, dash) if dash else "-"

    def plot_line(x: Any, y: Any, *, width: float, alpha: float) -> None:
        if halo:
            ax.plot(
                x,
                y,
                color="black",
                linewidth=width + 1.8,
                linestyle=linestyle,
                alpha=min(0.72, alpha * 0.80),
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=8,
            )
        ax.plot(
            x,
            y,
            color=colour,
            linewidth=width,
            linestyle=linestyle,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=9,
        )

    if isinstance(geometry, Point):
        ax.plot(
            geometry.x,
            geometry.y,
            marker="o",
            markersize=4.8,
            linestyle="None",
            markerfacecolor=colour,
            markeredgecolor="black",
            markeredgewidth=0.9,
            alpha=line_alpha,
            zorder=10,
        )
    elif isinstance(geometry, MultiPoint):
        for item in geometry.geoms:
            _plot_geometry(ax, item, style=style, linewidth=linewidth, halo=halo)
    elif isinstance(geometry, (LineString, MultiLineString)):
        geometries = geometry.geoms if isinstance(geometry, MultiLineString) else (geometry,)
        for item in geometries:
            x, y = item.xy
            plot_line(x, y, width=max(1.7, linewidth), alpha=line_alpha)
    elif isinstance(geometry, (Polygon, MultiPolygon)):
        polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else (geometry,)
        for polygon in polygons:
            x, y = polygon.exterior.xy
            ax.fill(x, y, color=colour, alpha=fill_alpha, zorder=6)
            plot_line(x, y, width=max(1.7, linewidth), alpha=line_alpha)
            for interior in polygon.interiors:
                x_hole, y_hole = interior.xy
                plot_line(x_hole, y_hole, width=max(1.2, linewidth * 0.8), alpha=line_alpha)
    elif isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            _plot_geometry(ax, item, style=style, linewidth=linewidth, halo=halo)


def _safe_render_geometry(geometry_data: Any, render_box: Any) -> Any | None:
    """Create, repair and clip one GeoJSON geometry for raster rendering.

    Historic England geometry is authoritative data, but occasional topology
    defects can make a direct Shapely intersection fail. The HTML renderer has
    historically fallen back to the original geometry in that case; the PNG
    renderer must be equally forgiving so a bad ring cannot silently make a
    monument disappear from the generated image.
    """
    from shapely.validation import make_valid

    if not geometry_data:
        return None
    try:
        geometry = shape(geometry_data)
    except Exception:
        return None
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        try:
            geometry = make_valid(geometry)
        except Exception:
            try:
                geometry = geometry.buffer(0)
            except Exception:
                return None
    if geometry.is_empty:
        return None
    try:
        clipped = geometry.intersection(render_box)
    except Exception:
        try:
            clipped = geometry.buffer(0).intersection(render_box)
        except Exception:
            return None
    if not clipped.is_valid:
        try:
            clipped = make_valid(clipped)
        except Exception:
            pass
    return clipped if not clipped.is_empty else None


def _draw_scale_bar(ax: Any, length_m: float) -> None:
    import matplotlib.patches as patches

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width = xmax - xmin
    height = ymax - ymin
    x = xmin + width * 0.05
    y = ymin + height * 0.05
    bar_height = height * 0.012
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            length_m,
            bar_height,
            facecolor="black",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            zorder=20,
        )
    )
    ax.text(
        x + length_m / 2,
        y + height * 0.016,
        f"{length_m:g} m",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
        zorder=21,
    )


def _elevation_tint(elevation: Any) -> tuple[Any, float, float]:
    """Return a restrained hypsometric tint and robust elevation bounds."""
    import numpy as np

    valid = np.asarray(elevation)[np.isfinite(elevation)]
    if valid.size == 0:
        return np.asarray(elevation), 0.0, 1.0
    low, high = np.nanpercentile(valid, (2.0, 98.0))
    if not high > low:
        high = low + 1.0
    return elevation, float(low), float(high)


def _render_extent_from_raster(dataset: Any) -> tuple[float, float, float, float]:
    return (
        float(dataset.bounds.left),
        float(dataset.bounds.bottom),
        float(dataset.bounds.right),
        float(dataset.bounds.top),
    )


def _read_lidar_arrays(dtm_path: Path) -> tuple[Any, Any, Any, tuple[float, float, float, float]]:
    import numpy as np
    import rasterio
    from prospector.terrain.anomalies import calculate_archaeological_hillshade, calculate_local_relief_models

    with rasterio.open(dtm_path) as dtm:
        if dtm.crs is None or dtm.crs.to_epsg() != 27700:
            raise ValueError(f"LiDAR raster has unexpected CRS: {dtm.crs}")
        elevation = dtm.read(1, masked=True).filled(np.nan)
        hillshade_array = calculate_archaeological_hillshade(elevation, dtm.res[0], dtm.res[1])
        reliefs = calculate_local_relief_models(elevation, dtm.res[0], dtm.res[1])
        lrm = max(reliefs.values(), key=lambda item: float(np.nanpercentile(np.abs(item), 99.5)))
        bounds = (float(dtm.bounds.left), float(dtm.bounds.bottom), float(dtm.bounds.right), float(dtm.bounds.top))
    return elevation, hillshade_array, lrm, bounds


def create_lidar_base_overlay(
    dtm_path: Path,
    output_path: Path,
    *,
    anomaly_base: bool = False,
) -> Path:
    """Create a clean LiDAR base image for HTML layer switching."""
    require_plotting_stack()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from prospector.reporting.overlay import _elevation_tint

    _elevation, hillshade_array, lrm, bounds = _read_lidar_arrays(dtm_path)
    display_extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    limit = max(float(np.nanpercentile(np.abs(lrm), 99.0)), 0.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    try:
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(hillshade_array, extent=display_extent, origin="upper", cmap="gray", vmin=0, vmax=255, interpolation="bilinear", zorder=1)
        if anomaly_base:
            ax.imshow(lrm, extent=display_extent, origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit, alpha=0.30, interpolation="bilinear", zorder=2)
        else:
            tint, tint_low, tint_high = _elevation_tint(_elevation)
            ax.imshow(tint, extent=display_extent, origin="upper", cmap="terrain", vmin=tint_low, vmax=tint_high, alpha=0.14, interpolation="bilinear", zorder=2)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches=None, facecolor="white", pad_inches=0)
    finally:
        plt.close(fig)
    return output_path


def create_lidar_aim_overlay(
    dtm_path: Path,
    aim_features: list[dict[str, Any]],
    study_area_wkt: str,
    centre_easting: float,
    centre_northing: float,
    output_path: Path,
    *,
    monument_extents: list[dict[str, Any]] | None = None,
    project_areas: list[dict[str, Any]] | None = None,
    hillshade_path: Path | None = None,
    render_bounds: tuple[float, float, float, float] | None = None,
    title: str = "Prospector — LiDAR / Historic England AIM",
) -> Path:
    """Create a high-contrast LiDAR map with semi-transparent HE AIM symbology."""
    del title  # Retained for API compatibility; map titles are handled by the report.
    require_plotting_stack()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    from prospector.terrain.anomalies import calculate_archaeological_hillshade

    output_path.parent.mkdir(parents=True, exist_ok=True)
    study_area = wkt.loads(study_area_wkt)
    monument_extents = monument_extents or []
    project_areas = project_areas or []

    with rasterio.open(dtm_path) as dtm:
        if dtm.crs is None or dtm.crs.to_epsg() != 27700:
            raise ValueError(f"LiDAR raster has unexpected CRS: {dtm.crs}")
        elevation = dtm.read(1, masked=True)
        raster_bounds = _render_extent_from_raster(dtm)
        display_bounds = render_bounds or raster_bounds
        if display_bounds != raster_bounds:
            # The current pipeline intentionally renders the full raster so that
            # every generated map uses the exact same pixel-to-coordinate extent.
            # Keep this guard informative if another caller tries to request a crop.
            if not (
                raster_bounds[0] <= display_bounds[0] <= display_bounds[2] <= raster_bounds[2]
                and raster_bounds[1] <= display_bounds[1] <= display_bounds[3] <= raster_bounds[3]
            ):
                raise ValueError("Requested render bounds do not lie inside the LiDAR raster")
            display_bounds = raster_bounds

        hillshade_array = calculate_archaeological_hillshade(
            elevation.filled(np.nan), dtm.res[0], dtm.res[1]
        )

    render_box = box(*display_bounds)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    try:
        fig.subplots_adjust(0, 0, 1, 1)
        display_extent = (display_bounds[0], display_bounds[2], display_bounds[1], display_bounds[3])
        ax.imshow(
            hillshade_array,
            extent=display_extent,
            origin="upper",
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="bilinear",
            zorder=1,
        )
        tint, tint_low, tint_high = _elevation_tint(elevation.filled(np.nan))
        ax.imshow(
            tint,
            extent=display_extent,
            origin="upper",
            cmap="terrain",
            vmin=tint_low,
            vmax=tint_high,
            alpha=0.14,
            interpolation="bilinear",
            zorder=2,
        )

        # Match AAME display order: project areas behind monument extents, with detailed
        # feature symbology drawn on top. The LiDAR renderer itself is unchanged.
        for feature in project_areas:
            geometry = _safe_render_geometry(feature.get("geometry"), render_box)
            if geometry is None:
                continue
            _plot_geometry(
                ax, geometry,
                style=HE_CONTEXT_STYLES["Project_Area"],
                linewidth=2.2,
            )

        for feature in monument_extents:
            geometry = _safe_render_geometry(feature.get("geometry"), render_box)
            if geometry is None:
                continue
            _plot_geometry(
                ax, geometry,
                style=HE_CONTEXT_STYLES["Monument_Extents"],
                linewidth=2.1,
            )

        for feature in aim_features:
            geometry = _safe_render_geometry(feature.get("geometry"), render_box)
            if geometry is None:
                continue
            properties = feature.get("properties") or {}
            style = he_style(
                properties.get("LAYER"),
                properties.get("MONUMENT_TYPE"),
                properties.get("EVIDENCE_1"),
            )
            _plot_geometry(ax, geometry, style=style, linewidth=2.1)

        ax.plot(
            centre_easting,
            centre_northing,
            marker="+",
            markersize=14,
            markeredgewidth=2.0,
            color="white",
            linestyle="None",
            zorder=20,
        )
        study_outline = study_area.intersection(render_box)
        _plot_geometry(
            ax,
            study_outline,
            style={"colour": "white", "line_alpha": 0.9, "fill_alpha": 0.0, "dash": None},
            linewidth=1.1,
        )

        ax.set_xlim(display_bounds[0], display_bounds[2])
        ax.set_ylim(display_bounds[1], display_bounds[3])
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches=None, facecolor="white", pad_inches=0)
    finally:
        plt.close(fig)
    return output_path



def _draw_candidate_overlays(ax: Any, candidates: list[Any]) -> None:
    for index, candidate in enumerate(candidates, 1):
        colour = "#F6E84A" if candidate.polarity == "positive" else "#46E0D0"
        _plot_geometry(
            ax,
            candidate.geometry,
            style={"colour": colour, "line_alpha": 0.90, "fill_alpha": 0.10, "dash": None},
            linewidth=1.5,
        )
        point = candidate.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            str(index),
            fontsize=7,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
                "pad": 1.5,
            },
            zorder=20,
        )


def create_lidar_anomaly_overlay(
    dtm_path: Path,
    candidates: list[Any],
    output_path: Path,
    *,
    hillshade_path: Path | None = None,
) -> Path:
    """Create a high-contrast LiDAR-only visualisation with terrain anomalies."""
    del hillshade_path  # v0.2.1 intentionally uses the enhanced renderer for consistency.
    require_plotting_stack()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    from prospector.terrain.anomalies import (
        calculate_archaeological_hillshade,
        calculate_local_relief_models,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dtm_path) as dtm:
        elevation = dtm.read(1, masked=True)
        data = elevation.filled(np.nan)
        hillshade_array = calculate_archaeological_hillshade(data, dtm.res[0], dtm.res[1])
        reliefs = calculate_local_relief_models(data, dtm.res[0], dtm.res[1])
        lrm = max(reliefs.values(), key=lambda item: float(np.nanpercentile(np.abs(item), 99.5)))
        map_extent = (
            float(dtm.bounds.left),
            float(dtm.bounds.right),
            float(dtm.bounds.bottom),
            float(dtm.bounds.top),
        )

    limit = float(np.nanpercentile(np.abs(lrm), 99.0))
    limit = max(limit, 0.05)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    try:
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(
            hillshade_array,
            extent=map_extent,
            origin="upper",
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="bilinear",
            zorder=1,
        )
        ax.imshow(
            lrm,
            extent=map_extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            alpha=0.34,
            interpolation="bilinear",
            zorder=2,
        )
        _draw_candidate_overlays(ax, candidates)
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches=None, facecolor="white", pad_inches=0)
    finally:
        plt.close(fig)
    return output_path


def create_lidar_anomaly_aim_overlay(
    dtm_path: Path,
    candidates: list[Any],
    aim_features: list[dict[str, Any]],
    output_path: Path,
    *,
    monument_extents: list[dict[str, Any]] | None = None,
    project_areas: list[dict[str, Any]] | None = None,
) -> Path:
    """Create the review image combining terrain anomalies and known AIM features."""
    require_plotting_stack()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from prospector.terrain.anomalies import (
        calculate_archaeological_hillshade,
        calculate_local_relief_models,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    monument_extents = monument_extents or []
    project_areas = project_areas or []
    with rasterio.open(dtm_path) as dtm:
        elevation = dtm.read(1, masked=True)
        data = elevation.filled(np.nan)
        hillshade_array = calculate_archaeological_hillshade(data, dtm.res[0], dtm.res[1])
        reliefs = calculate_local_relief_models(data, dtm.res[0], dtm.res[1])
        lrm = max(reliefs.values(), key=lambda item: float(np.nanpercentile(np.abs(item), 99.5)))
        map_bounds = (
            float(dtm.bounds.left),
            float(dtm.bounds.bottom),
            float(dtm.bounds.right),
            float(dtm.bounds.top),
        )
        display_extent = (map_bounds[0], map_bounds[2], map_bounds[1], map_bounds[3])

    limit = max(float(np.nanpercentile(np.abs(lrm), 99.0)), 0.05)
    render_box = box(*map_bounds)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    try:
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(
            hillshade_array,
            extent=display_extent,
            origin="upper",
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="bilinear",
            zorder=1,
        )
        ax.imshow(
            lrm,
            extent=display_extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            alpha=0.28,
            interpolation="bilinear",
            zorder=2,
        )
        render_box = box(*map_bounds)
        for feature in project_areas:
            geometry = _safe_render_geometry(feature.get("geometry"), render_box)
            if geometry is None:
                continue
            _plot_geometry(ax, geometry, style=HE_CONTEXT_STYLES["Project_Area"], linewidth=2.2)

        for feature in monument_extents:
            geometry = _safe_render_geometry(feature.get("geometry"), render_box)
            if geometry is None:
                continue
            _plot_geometry(ax, geometry, style=HE_CONTEXT_STYLES["Monument_Extents"], linewidth=2.1)

        for feature in aim_features:
            geometry = _safe_render_geometry(feature.get("geometry"), render_box)
            if geometry is None:
                continue
            properties = feature.get("properties") or {}
            style = he_style(
                properties.get("LAYER"),
                properties.get("MONUMENT_TYPE"),
                properties.get("EVIDENCE_1"),
            )
            _plot_geometry(ax, geometry, style=style, linewidth=1.95)


        # Candidate outlines and labels remain on top of the Historic England layers.
        _draw_candidate_overlays(ax, candidates)

        ax.set_xlim(map_bounds[0], map_bounds[2])
        ax.set_ylim(map_bounds[1], map_bounds[3])
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches=None, facecolor="white", pad_inches=0)
    finally:
        plt.close(fig)
    return output_path
