from __future__ import annotations

import base64
import html
import json
import os
import re
import struct
from pathlib import Path
from typing import Any

from shapely.geometry import box, shape

from prospector.reporting.overlay import HE_CONTEXT_STYLES, HE_LAYER_STYLES, he_style


def _escape(value: object) -> str:
    return html.escape(str(value))


def _relative_link(destination: Path, target: Path | None) -> str:
    if target is None or not target.is_file():
        return "Not available"
    relative = Path(os.path.relpath(target.resolve(), destination.parent.resolve()))
    return f'<a href="{html.escape(relative.as_posix())}">{html.escape(target.name)}</a>'


def _relative_src(destination: Path, target: Path | None) -> str | None:
    if target is None or not target.is_file():
        return None
    relative = Path(os.path.relpath(target.resolve(), destination.parent.resolve()))
    return html.escape(relative.as_posix(), quote=True)


def _first(properties: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = properties.get(key)
        if value not in (None, "", "-"):
            return str(value)
    return ""


def _urls(properties: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key, value in properties.items():
        key_normalised = str(key).casefold().replace("-", "_")
        if not (key_normalised.startswith("he_url") or key_normalised.startswith("her_link")):
            continue
        if not value:
            continue
        for url in str(value).split():
            if url.startswith(("http://", "https://")) and url not in urls:
                urls.append(url)
    return urls


def _record_key(feature: dict[str, Any], index: int) -> str:
    properties = feature.get("properties") or {}
    return _first(properties, "HE_UID", "DHEUID_1", "HER_NO", "HERNO_1", "OBJECTID") or f"feature-{index}"


def _group_aim_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(features, 1):
        properties = feature.get("properties") or {}
        key = _record_key(feature, index)
        group = groups.setdefault(
            key,
            {
                "key": key,
                "features": [],
                "types": set(),
                "periods": set(),
                "evidence": set(),
                "sources": set(),
                "projects": set(),
                "her_numbers": set(),
                "urls": [],
            },
        )
        group["features"].append(feature)
        for field, target in (
            ("MONUMENT_TYPE", "types"),
            ("PERIOD", "periods"),
            ("EVIDENCE_1", "evidence"),
            ("SOURCE_1", "sources"),
            ("PROJECT", "projects"),
            ("project", "projects"),
        ):
            value = properties.get(field)
            if value:
                group[target].add(str(value))
        for field in ("HER_NO", "HERNO_1", "HERNO_2", "HERNO_3", "HERNO_4", "HERNO_5"):
            value = properties.get(field)
            if value:
                group["her_numbers"].add(str(value))
        for url in _urls(properties):
            if url not in group["urls"]:
                group["urls"].append(url)

    result = []
    for group in groups.values():
        for field in ("types", "periods", "evidence", "sources", "projects", "her_numbers"):
            group[field] = sorted(group[field])
        result.append(group)
    return result


def _format_raw_properties(properties: dict[str, Any]) -> str:
    return html.escape(json.dumps(properties, indent=2, default=str, sort_keys=True))


def _clip_geometry(geometry: Any, study_bounds: tuple[float, float, float, float]) -> Any:
    """Clip a map geometry to the exact image rectangle used by the report."""
    if geometry.is_empty:
        return geometry
    try:
        if not geometry.is_valid:
            from shapely.validation import make_valid
            geometry = make_valid(geometry)
        return geometry.intersection(box(*study_bounds))
    except Exception:
        try:
            return geometry.buffer(0).intersection(box(*study_bounds))
        except Exception:
            return geometry


def _svg_path(
    geometry: Any,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> str:
    x_span = xmax - xmin
    y_span = ymax - ymin
    if not x_span > 0 or not y_span > 0 or width <= 0 or height <= 0:
        return ""

    def p(x: float, y: float) -> str:
        px = min(width, max(0.0, (x - xmin) / x_span * width))
        py = min(height, max(0.0, (ymax - y) / y_span * height))
        return f"{px:.2f},{py:.2f}"

    def ring(coords: Any) -> str:
        points = list(coords)
        if not points:
            return ""
        return "M " + " L ".join(p(x, y) for x, y in points) + " Z"

    if geometry.is_empty:
        return ""
    geometry_type = geometry.geom_type
    if geometry_type == "Point":
        x, y = p(geometry.x, geometry.y).split(",")
        return f"M {x},{y} m -4,0 a 4,4 0 1,0 8,0 a 4,4 0 1,0 -8,0"
    if geometry_type == "LineString":
        points = list(geometry.coords)
        return "M " + " L ".join(p(x, y) for x, y in points)
    if geometry_type == "Polygon":
        paths = [ring(geometry.exterior.coords)]
        paths.extend(ring(interior.coords) for interior in geometry.interiors)
        return " ".join(path for path in paths if path)
    if geometry_type.startswith("Multi"):
        return " ".join(_svg_path(part, xmin, ymin, xmax, ymax, width, height) for part in geometry.geoms)
    if geometry_type == "GeometryCollection":
        return " ".join(_svg_path(part, xmin, ymin, xmax, ymax, width, height) for part in geometry.geoms)
    return ""


def _feature_svg(
    features: list[dict[str, Any]],
    study_bounds: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> str:
    xmin, ymin, xmax, ymax = study_bounds
    groups = _group_aim_features(features)
    group_index_by_key = {group["key"]: index for index, group in enumerate(groups, 1)}
    elements: list[str] = []
    for index, feature in enumerate(features, 1):
        geometry_data = feature.get("geometry")
        if not geometry_data:
            continue
        try:
            geometry = _clip_geometry(shape(geometry_data), study_bounds)
        except Exception:
            continue
        if geometry.is_empty:
            continue
        properties = feature.get("properties") or {}
        path = _svg_path(geometry, xmin, ymin, xmax, ymax, image_width, image_height)
        if not path:
            continue
        style = he_style(
            properties.get("LAYER"),
            properties.get("MONUMENT_TYPE"),
            properties.get("EVIDENCE_1"),
        )
        colour = style["colour"]
        urls = _urls(properties)
        group_key = _record_key(feature, index)
        group_index = group_index_by_key.get(group_key, index)
        href = urls[0] if urls else f"#he-feature-{group_index}"
        target = ' target="_blank" rel="noopener"' if urls else ""
        tooltip = " | ".join(
            filter(
                None,
                [
                    _first(properties, "MONUMENT_TYPE"),
                    _first(properties, "PERIOD"),
                    _first(properties, "EVIDENCE_1"),
                    _first(properties, "HE_UID"),
                ],
            )
        )
        stroke_alpha = float(style.get("line_alpha", 1.0))
        elements.append(
            f'<a href="{html.escape(href)}"{target}>'
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-opacity="{stroke_alpha:.2f}" '
            f'stroke-width="3" vector-effect="non-scaling-stroke" pointer-events="stroke">'
            f'<title>{_escape(tooltip or f"Historic England feature {index}")}</title></path></a>'
        )
    return "".join(elements)


def _map_viewport(
    image_width: int,
    image_height: int,
    map_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float, int, int]:
    """Return the active equal-aspect map rectangle inside the PNG.

    The overlays are projected in map coordinates, while the PNGs are rendered
    on 12x9 figures with matplotlib's equal aspect. That leaves letterboxing
    around the square 1 km data extent. The HTML SVG must occupy that same
    rectangle instead of the full PNG, otherwise the vectors stretch sideways.
    """
    xmin, ymin, xmax, ymax = map_bounds
    x_span = max(1e-9, xmax - xmin)
    y_span = max(1e-9, ymax - ymin)
    map_aspect = x_span / y_span
    image_aspect = image_width / image_height
    if map_aspect >= image_aspect:
        viewport_width = image_width
        viewport_height = max(1, round(image_width / map_aspect))
        left = 0.0
        top = (image_height - viewport_height) / 2.0
    else:
        viewport_height = image_height
        viewport_width = max(1, round(image_height * map_aspect))
        left = (image_width - viewport_width) / 2.0
        top = 0.0
    return (
        left / image_width * 100.0,
        top / image_height * 100.0,
        viewport_width / image_width * 100.0,
        viewport_height / image_height * 100.0,
        viewport_width,
        viewport_height,
    )


def _interactive_map(
    image_data: str,
    alt: str,
    overlay_svg: str,
    width: int,
    height: int,
    *,
    viewport: tuple[float, float, float, float, int, int],
    extra_class: str = "",
    clip_id: str = "map",
) -> str:
    safe_clip_id = re.sub(r"[^a-zA-Z0-9_-]", "-", clip_id)
    left, top, viewport_width_pct, viewport_height_pct, viewport_width, viewport_height = viewport
    svg_style = (
        f"left:{left:.6f}%;top:{top:.6f}%;"
        f"width:{viewport_width_pct:.6f}%;height:{viewport_height_pct:.6f}%;"
    )
    return f'''
<div class="interactive-map {extra_class}" style="aspect-ratio:{width}/{height}">
<img src="{image_data}" alt="{html.escape(alt)}">
<svg style="{svg_style}" viewBox="0 0 {viewport_width} {viewport_height}" preserveAspectRatio="none" aria-hidden="true">
<defs><clipPath id="clip-{safe_clip_id}"><rect x="0" y="0" width="{viewport_width}" height="{viewport_height}"></rect></clipPath></defs>
<g clip-path="url(#clip-{safe_clip_id})">{overlay_svg}</g>
</svg>
</div>
'''


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected a PNG image")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions are invalid")
    return width, height


def _image_data(path: Path) -> tuple[str, int, int]:
    data = path.read_bytes()
    width, height = _png_dimensions(data)
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii"), width, height


def _he_legend() -> str:
    rows = []
    for name, style in HE_CONTEXT_STYLES.items():
        dash = "border-top: 3px dashed" if style.get("dash") else "border-top: 3px solid"
        rows.append(
            f'<span class="legend-swatch" style="{dash} {style["colour"]}"></span>'
            f'<span><strong>{_escape(name)}</strong> — {_escape(style.get("description", ""))}</span>'
        )
    for name, style in HE_LAYER_STYLES.items():
        dash = "border-top: 3px dashed" if style.get("dash") else "border-top: 3px solid"
        rows.append(
            f'<span class="legend-swatch" style="{dash} {style["colour"]}"></span>'
            f'<span><strong>{_escape(name)}</strong> — {_escape(style.get("description", ""))}</span>'
        )
    return '<div class="legend">' + "".join(rows) + "</div>"


def _aim_record_rows(features: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for index, group in enumerate(_group_aim_features(features), 1):
        anchor = f"he-feature-{index}"
        urls = group["urls"]
        record = (
            " · ".join(
                f'<a href="{html.escape(url)}" target="_blank" rel="noopener">Heritage Gateway record</a>'
                for url in urls[:2]
            )
            or "No record URL supplied"
        )
        rows.append(
            f'<tr id="{anchor}"><td>{index}</td><td><strong>{_escape(group["key"])}</strong><br>{record}</td>'
            f'<td>{_escape(", ".join(group["types"]) or "—")}</td>'
            f'<td>{_escape(", ".join(group["periods"]) or "—")}</td>'
            f'<td>{_escape(", ".join(group["evidence"]) or "—")}</td>'
            f'<td>{_escape(", ".join(group["sources"]) or "—")}</td>'
            f'<td>{len(group["features"])}</td>'
            f'<td><details><summary>Raw attributes</summary><pre>{_format_raw_properties(group["features"][0].get("properties") or {})}</pre></details></td></tr>'
        )
    return "".join(rows)


def _candidate_svg(
    candidates: list[Any],
    study_bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    """Render clickable anomaly geometry plus an always-visible numbered marker."""
    xmin, ymin, xmax, ymax = study_bounds
    elements: list[str] = []
    x_span = max(1e-9, xmax - xmin)
    y_span = max(1e-9, ymax - ymin)

    def project(x: float, y: float) -> tuple[float, float]:
        return (
            min(width, max(0.0, (x - xmin) / x_span * width)),
            min(height, max(0.0, (ymax - y) / y_span * height)),
        )

    for index, candidate in enumerate(candidates, 1):
        geometry = _clip_geometry(candidate.geometry, study_bounds)
        if geometry.is_empty:
            continue
        path = _svg_path(geometry, xmin, ymin, xmax, ymax, width, height)
        point = geometry.representative_point()
        marker_x, marker_y = project(point.x, point.y)
        if not path:
            path = f"M {marker_x:.2f},{marker_y:.2f}"
        colour = "#F6E84A" if candidate.polarity == "positive" else "#46E0D0"
        tooltip = (
            f"Candidate {index} — {candidate.classification}; "
            f"score {candidate.score:.1f}; {candidate.polarity}; "
            f"{candidate.relief_m:.2f} m relief; {candidate.area_m2:.1f} m²"
        )
        elements.append(
            f'<g id="map-candidate-{index}" data-map-candidate="{index}" class="candidate-marker">'
            f'<a href="#candidate-{index}" data-candidate-map-link="{index}">'
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-opacity="0.92" stroke-width="3.2" '
            f'vector-effect="non-scaling-stroke" pointer-events="stroke">'
            f'<title>{_escape(tooltip)}</title></path>'
            f'<circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="9" fill="rgba(0,0,0,.82)" stroke="{colour}" stroke-width="2" vector-effect="non-scaling-stroke"></circle>'
            f'<text x="{marker_x:.2f}" y="{marker_y + 4:.2f}" text-anchor="middle" font-size="9" font-weight="700" '
            f'font-family="system-ui, sans-serif" fill="white" pointer-events="none">{index}</text>'
            f'</a></g>'
        )
    return "".join(elements)

def _layered_map(
    destination: Path,
    *,
    bases: list[tuple[str, str, Path | None]],
    he_svg: str,
    candidate_svg: str,
    width: int,
    height: int,
    viewport: tuple[float, float, float, float, int, int],
) -> str:
    """Build a layered map whose controls sit above, never over, the image."""
    available = [(key, label, _relative_src(destination, path)) for key, label, path in bases]
    available = [(key, label, src) for key, label, src in available if src]
    if not available:
        return '<p class="empty">No switchable base imagery was generated for this run.</p>'

    left, top, width_pct, height_pct, viewport_width, viewport_height = viewport
    images = []
    buttons = []
    for index, (key, label, src) in enumerate(available):
        active = index == 0
        display = "block" if active else "none"
        images.append(
            f'<img class="layer-base" data-base="{_escape(key)}" src="{src}" alt="{html.escape(label)}" style="display:{display}">'
        )
        buttons.append(
            f'<button type="button" class="base-button{" active" if active else ""}" data-base-button="{_escape(key)}">{_escape(label)}</button>'
        )

    svg_style = f'left:{left:.6f}%;top:{top:.6f}%;width:{width_pct:.6f}%;height:{height_pct:.6f}%;'
    return f"""
<div class="layer-map">
  <details class="layer-controls" open>
    <summary><span class="menu-glyph">☰</span> Layers &amp; base imagery</summary>
    <div class="layer-control-body">
      <div class="control-group"><strong>Base</strong>{"".join(buttons)}</div>
      <div class="control-group"><strong>Overlays</strong><label><input type="checkbox" data-toggle-overlay="he" checked> Historic England</label>
      <label><input type="checkbox" data-toggle-overlay="candidates" checked> Anomaly candidates</label></div>
    </div>
  </details>
  <div class="layer-canvas" style="aspect-ratio:{width}/{height}">
    {"".join(images)}
    <svg class="layer-svg" style="{svg_style}" viewBox="0 0 {viewport_width} {viewport_height}" preserveAspectRatio="none" aria-hidden="true">
      <g data-overlay="he">{he_svg}</g><g data-overlay="candidates">{candidate_svg}</g>
    </svg>
  </div>
</div>
"""


def write_html_report(
    destination: Path,
    *,
    run_id: str,
    application_version: str,
    latitude: float,
    longitude: float,
    easting: float,
    northing: float,
    diameter_m: float,
    aim_features: list[dict[str, Any]],
    monument_extents: list[dict[str, Any]] | None = None,
    project_areas: list[dict[str, Any]] | None = None,
    cache_entries: list[dict[str, Any]],
    errors: list[str],
    overlay_path: Path | None,
    dtm_path: Path | None,
    hillshade_path: Path | None,
    aim_geojson_path: Path | None,
    anomaly_overlay_path: Path | None = None,
    anomaly_aim_overlay_path: Path | None = None,
    candidates: list[Any] | None = None,
    candidate_geojson_path: Path | None = None,
    anomaly_metadata: dict[str, Any] | None = None,
    study_bounds: tuple[float, float, float, float] | None = None,
    map_bounds: tuple[float, float, float, float] | None = None,
    modern_context_path: Path | None = None,
    modern_context_raster_path: Path | None = None,
    satellite_preview_path: Path | None = None,
    satellite_support_path: Path | None = None,
    satellite_search_path: Path | None = None,
    lidar_base_path: Path | None = None,
    anomaly_base_path: Path | None = None,
    location_name: str | None = None,
    location_display_name: str | None = None,
    location_attribution: str | None = None,
) -> Path:
    'Write the V0.4.1 report with first-class maps, layered evidence and detector provenance.'
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = candidates or []
    monument_extents = monument_extents or []
    project_areas = project_areas or []
    study_bounds = study_bounds or (
        easting - diameter_m / 2,
        northing - diameter_m / 2,
        easting + diameter_m / 2,
        northing + diameter_m / 2,
    )
    map_bounds = map_bounds or study_bounds

    overlay_html = '<p class="empty">No LiDAR/Historic England map was generated for this run.</p>'
    if overlay_path is not None and overlay_path.is_file():
        image_data, map_width, map_height = _image_data(overlay_path)
        viewport = _map_viewport(map_width, map_height, map_bounds)
        _, _, _, _, viewport_width, viewport_height = viewport
        aim_visual_features = project_areas + monument_extents + aim_features
        overlay_svg = _feature_svg(aim_visual_features, map_bounds, viewport_width, viewport_height)
        overlay_html = _interactive_map(
            image_data,
            "LiDAR hillshade with Historic England Aerial Investigation and Mapping features",
            overlay_svg,
            map_width,
            map_height,
            viewport=viewport,
            clip_id="aim",
        )

    anomaly_map_html = '<p class="empty">No LiDAR anomaly map was generated for this run.</p>'
    if anomaly_overlay_path is not None and anomaly_overlay_path.is_file():
        anomaly_data, anomaly_width, anomaly_height = _image_data(anomaly_overlay_path)
        viewport = _map_viewport(anomaly_width, anomaly_height, map_bounds)
        _, _, _, _, viewport_width, viewport_height = viewport
        anomaly_svg = _candidate_svg(candidates, map_bounds, viewport_width, viewport_height)
        anomaly_map_html = _interactive_map(
            anomaly_data,
            "LiDAR terrain anomaly visualisation",
            anomaly_svg,
            anomaly_width,
            anomaly_height,
            viewport=viewport,
            extra_class="candidate-map",
            clip_id="candidates",
        )

    combined_map_html = '<p class="empty">No combined anomalies + Historic England map was generated for this run.</p>'
    if anomaly_aim_overlay_path is not None and anomaly_aim_overlay_path.is_file():
        combined_data, combined_width, combined_height = _image_data(anomaly_aim_overlay_path)
        combined_map_html = (
            f'<div class="map-image" style="aspect-ratio:{combined_width}/{combined_height}">'
            f'<img src="{combined_data}" alt="LiDAR terrain anomalies with Historic England features">'
            f'</div>'
        )

    satellite_html = '<p class="empty">No satellite context image was generated for this run.</p>'
    if satellite_preview_path is not None and satellite_preview_path.is_file():
        satellite_data, satellite_width, satellite_height = _image_data(satellite_preview_path)
        satellite_html = (
            f'<div class="map-image" style="aspect-ratio:{satellite_width}/{satellite_height}">'
            f'<img src="{satellite_data}" alt="Latest Sentinel-2 satellite context">'
            f'</div>'
        )

    error_html = (
        '<div class="warning"><strong>Acquisition / analysis warnings</strong><ul>'
        + "".join(f"<li>{_escape(error)}</li>" for error in errors)
        + "</ul></div>"
        if errors
        else '<p class="ok">No acquisition or analysis errors were recorded.</p>'
    )
    cache_hits = sum(1 for item in cache_entries if item.get("cache_hit"))
    downloads = len(cache_entries) - cache_hits

    candidate_rows = "".join(
        f'<tr id="candidate-{i}" class="candidate-row">'
        f'<td><a class="candidate-number" href="#map-candidate-{i}" data-focus-candidate="{i}">{i}</a></td>'
        f'<td><a class="view-link" href="#map-candidate-{i}" data-focus-candidate="{i}">View on map ↗</a></td>'
        f'<td><strong>{_escape(c.classification)}</strong></td>'
        f'<td>{c.score:.1f}</td><td>{c.lidar_score:.1f}</td><td>{c.persistence_score:.1f}</td>'
        f'<td>{c.morphology_score:.1f}</td><td>{c.linear_score:.1f}</td><td>{c.ring_score:.1f}</td>'
        f'<td>{c.ridge_valley_score:.1f}</td><td>{c.terrain_novelty_score:.1f}</td><td>{c.texture_score:.1f}</td>'
        f'<td>{c.modern_penalty:.1f}</td><td>{c.satellite_support:.1f}</td><td>{c.he_similarity:.1f}</td>'
        f'<td>{c.relief_m:.2f} m</td><td>{c.strongest_scale_m:g} m</td><td>{c.area_m2:,.1f} m²</td>'
        f'<td>{_escape("; ".join(c.reasons) or "—")}</td></tr>'
        for i, c in enumerate(candidates, 1)
    )
    metadata_json = _escape(json.dumps(anomaly_metadata or {}, indent=2, default=str, sort_keys=True))

    modern_summary = (anomaly_metadata or {}).get("external_context", {})
    context_types = []
    for key in ("os_openmap", "openstreetmap"):
        item = modern_summary.get(key) or {}
        for context_type in item.get("context_types", []) or []:
            if context_type not in context_types:
                context_types.append(context_type)
    modern_count = int(modern_summary.get("modern_feature_count", 0) or 0)
    satellite_meta = modern_summary.get("satellite") or (anomaly_metadata or {}).get("satellite_context", {}) or {}
    diagnostic_rasters = (anomaly_metadata or {}).get("diagnostic_rasters", {}) or {}
    diagnostic_links = "".join(
        f'<li>{_escape(label.replace("_", " ").title())}: {_relative_link(destination, Path(path))}</li>'
        for label, path in diagnostic_rasters.items()
        if path
    )
    diagnostic_html = (
        f'<details class="diagnostic-details"><summary>Open detector diagnostic layers</summary>'
        f'<p class="note">These rasters expose the individual discovery channels used to rank candidates. They are particularly useful when comparing sensitivity levels and investigating subtle features.</p>'
        f'<ul>{diagnostic_links}</ul></details>'
        if diagnostic_links else ""
    )

    interactive_map_html = '<p class="empty">No interactive layer map was generated for this run.</p>'
    map_reference = next(
        (path for path in (overlay_path, anomaly_overlay_path, lidar_base_path, anomaly_base_path) if path is not None and path.is_file()),
        None,
    )
    if map_reference is not None:
        _, map_width, map_height = _image_data(map_reference)
        viewport = _map_viewport(map_width, map_height, map_bounds)
        _, _, _, _, viewport_width, viewport_height = viewport
        he_visual_features = project_areas + monument_extents + aim_features
        he_svg = _feature_svg(he_visual_features, map_bounds, viewport_width, viewport_height)
        candidate_svg = _candidate_svg(candidates, map_bounds, viewport_width, viewport_height)
        interactive_map_html = _layered_map(
            destination,
            bases=[
                ("lidar", "LiDAR hillshade", lidar_base_path),
                ("anomalies", "LiDAR anomaly relief", anomaly_base_path),
                ("satellite", "Sentinel-2", satellite_preview_path),
            ],
            he_svg=he_svg,
            candidate_svg=candidate_svg,
            width=map_width,
            height=map_height,
            viewport=viewport,
        )

    sensitivity_value = (anomaly_metadata or {}).get("sensitivity", 5)
    try:
        sensitivity_level = int(sensitivity_value)
    except (TypeError, ValueError):
        sensitivity_level = 5
    sensitivity_name = f"{sensitivity_level} / 10"
    detector_workers = (anomaly_metadata or {}).get("workers_requested", 0)
    detector_workers_label = "auto" if detector_workers in (None, 0) else str(detector_workers)

    layer_script = """<script>
(function() {
  const root = document.documentElement;
  const savedTheme = window.localStorage ? localStorage.getItem('prospector-theme') : null;
  const preferredTheme = savedTheme || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  root.dataset.theme = preferredTheme;

  function setTheme(theme) {
    root.dataset.theme = theme;
    if (window.localStorage) localStorage.setItem('prospector-theme', theme);
  }
  document.querySelectorAll('[data-theme-choice]').forEach(function(button) {
    button.addEventListener('click', function() { setTheme(button.getAttribute('data-theme-choice')); });
  });
  document.querySelectorAll('[data-theme-toggle]').forEach(function(button) {
    button.addEventListener('click', function() { setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark'); });
  });

  document.querySelectorAll('.layer-map').forEach(function(map) {
    map.querySelectorAll('[data-base-button]').forEach(function(button) {
      button.addEventListener('click', function() {
        const key = button.getAttribute('data-base-button');
        map.querySelectorAll('.layer-base').forEach(function(image) {
          image.style.display = image.getAttribute('data-base') === key ? 'block' : 'none';
        });
        map.querySelectorAll('[data-base-button]').forEach(function(item) {
          item.classList.toggle('active', item === button);
        });
      });
    });
    map.querySelectorAll('[data-toggle-overlay]').forEach(function(input) {
      input.addEventListener('change', function() {
        const key = input.getAttribute('data-toggle-overlay');
        const group = map.querySelector('[data-overlay="' + key + '"]');
        if (group) group.style.display = input.checked ? 'block' : 'none';
      });
    });
  });

  function selectCandidate(index) {
    const marker = document.querySelector('[data-map-candidate="' + index + '"]');
    const row = document.getElementById('candidate-' + index);
    document.querySelectorAll('.candidate-marker.is-selected').forEach(function(item) { item.classList.remove('is-selected'); });
    document.querySelectorAll('tr.candidate-row.is-selected').forEach(function(item) { item.classList.remove('is-selected'); });
    if (marker) marker.classList.add('is-selected');
    if (row) row.classList.add('is-selected');
  }

  document.querySelectorAll('[data-focus-candidate]').forEach(function(element) {
    element.addEventListener('click', function(event) {
      event.preventDefault();
      const index = element.getAttribute('data-focus-candidate');
      selectCandidate(index);
      const marker = document.querySelector('[data-map-candidate="' + index + '"]');
      if (marker) marker.scrollIntoView({behavior:'smooth', block:'center'});
    });
  });
  document.querySelectorAll('[data-candidate-map-link]').forEach(function(element) {
    element.addEventListener('click', function(event) {
      event.preventDefault();
      const index = element.getAttribute('data-candidate-map-link');
      selectCandidate(index);
      const row = document.getElementById('candidate-' + index);
      if (row) row.scrollIntoView({behavior:'smooth', block:'center'});
    });
  });
})();
</script>
"""

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prospector — {_escape(run_id)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f7fb; --panel: #ffffff; --panel-2: #eef2f7; --text: #172033; --muted: #667085;
  --border: #d9e0ea; --accent: #5b63f6; --accent-soft: rgba(91,99,246,.12); --shadow: 0 14px 40px rgba(16,24,40,.08);
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #0c1017; --panel: #131a24; --panel-2: #1a2330; --text: #edf2f7; --muted: #9aa7b8;
  --border: #293444; --accent: #8b93ff; --accent-soft: rgba(139,147,255,.15); --shadow: 0 18px 50px rgba(0,0,0,.32);
}}
:root[data-theme="rose"] {{
  --bg:#fff7f8; --panel:#fffdfd; --panel-2:#fdecef; --text:#3f2f34; --muted:#826d74;
  --border:#edd5db; --accent:#bb7183; --accent-soft:rgba(187,113,131,.14); --shadow:0 14px 40px rgba(126,68,85,.10);
}}
:root[data-theme="lavender"] {{
  --bg:#faf8ff; --panel:#ffffff; --panel-2:#f0ecff; --text:#342d45; --muted:#756b8d;
  --border:#ddd6f2; --accent:#8170b8; --accent-soft:rgba(129,112,184,.14); --shadow:0 14px 40px rgba(79,61,126,.09);
}}
:root[data-theme="mint"] {{
  --bg:#f5fcf9; --panel:#ffffff; --panel-2:#e8f6f0; --text:#263c34; --muted:#668078;
  --border:#cde5da; --accent:#5b9d82; --accent-soft:rgba(91,157,130,.14); --shadow:0 14px 40px rgba(59,112,91,.09);
}}
:root[data-theme="sky"] {{
  --bg:#f5faff; --panel:#ffffff; --panel-2:#e9f3fb; --text:#263746; --muted:#667d8e;
  --border:#d1e3ef; --accent:#5f94b8; --accent-soft:rgba(95,148,184,.14); --shadow:0 14px 40px rgba(51,103,137,.09);
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:linear-gradient(180deg,var(--bg),var(--bg)); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.5; }}
main {{ max-width:1500px; margin:0 auto; padding:1.3rem 1.4rem 3rem; }}
.topbar {{ position:sticky; top:0; z-index:100; margin:0 -1.4rem 1.3rem; padding:.8rem 1.4rem; background:color-mix(in srgb,var(--bg) 88%, transparent); backdrop-filter:blur(18px); border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; gap:1rem; }}
.brand {{ display:flex; align-items:center; gap:.7rem; }}
.brand-mark {{ width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,var(--accent),#23c7d9); box-shadow:0 8px 24px rgba(91,99,246,.25); }}
.brand h1 {{ margin:0; font-size:1.05rem; letter-spacing:-.02em; }}
.theme-button,.base-button {{ border:1px solid var(--border); border-radius:10px; background:var(--panel); color:var(--text); padding:.45rem .7rem; cursor:pointer; transition:.15s ease; }}
.theme-button:hover,.base-button:hover {{ border-color:var(--accent); transform:translateY(-1px); }}
.hero {{ padding:1.2rem 0 .8rem; }}
.hero h2 {{ margin:.15rem 0 .2rem; font-size:clamp(1.55rem,3vw,2.4rem); letter-spacing:-.04em; }}
.hero p {{ color:var(--muted); margin:0; }}
section {{ margin:1.1rem 0; padding:1.1rem; background:var(--panel); border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow); }}
section h2 {{ margin-top:0; font-size:1.1rem; letter-spacing:-.02em; }}
img {{ max-width:100%; height:auto; display:block; }}
table {{ border-collapse:separate; border-spacing:0; width:100%; font-size:.88rem; }}
th,td {{ padding:.62rem .68rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
th {{ position:sticky; top:57px; z-index:10; background:var(--panel-2); color:var(--muted); font-size:.73rem; text-transform:uppercase; letter-spacing:.06em; }}
tbody tr:hover {{ background:var(--accent-soft); }}
.candidate-row.is-selected {{ background:var(--accent-soft); outline:2px solid var(--accent); outline-offset:-2px; }}
.candidate-number {{ font-weight:800; color:var(--accent); text-decoration:none; }}
.view-link {{ white-space:nowrap; color:var(--accent); text-decoration:none; font-weight:650; }}
.view-link:hover {{ text-decoration:underline; }}
pre {{ white-space:pre-wrap; max-height:400px; overflow:auto; background:var(--panel-2); border-radius:12px; padding:.9rem; }}
.warning {{ padding:1rem; background:#fff3cd; color:#5d4700; border-radius:12px; }}
:root[data-theme="dark"] .warning {{ background:#3a2f13; color:#f8e5a9; }}
.ok {{ padding:1rem; background:#e8f5e9; border-radius:12px; }}
.interactive-map,.map-image {{ position:relative; width:100%; overflow:hidden; background:#080b10; border-radius:14px; }}
.interactive-map img,.interactive-map svg,.map-image img {{ position:absolute; inset:0; width:100%; height:100%; display:block; }}
.interactive-map svg path {{ cursor:pointer; transition:stroke-width .12s, opacity .12s; }}
.interactive-map svg path:hover {{ stroke-width:6 !important; opacity:1; }}
.candidate-marker.is-selected path {{ stroke-width:7 !important; opacity:1; }}
.candidate-marker.is-selected circle {{ r:11; }}
.theme-menu summary {{ cursor:pointer; list-style:none; padding:.45rem .65rem; border:1px solid var(--border); border-radius:10px; background:var(--panel); color:var(--text); font-weight:700; }} .theme-menu summary::-webkit-details-marker {{ display:none; }}
.theme-options {{ position:absolute; right:0; top:calc(100% + .45rem); z-index:120; display:grid; grid-template-columns:1fr 1fr; gap:.35rem; min-width:190px; padding:.5rem; background:var(--panel); border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow); }}
.theme-options button {{ border:1px solid var(--border); background:var(--panel-2); color:var(--text); padding:.4rem .5rem; border-radius:8px; cursor:pointer; }}
.layer-map {{ position:relative; width:100%; background:#080b10; border-radius:14px; overflow:hidden; }}
.layer-controls {{ position:relative; z-index:30; margin:0 0 .55rem; padding:.1rem .75rem .45rem; background:var(--panel-2); color:var(--text); border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow); }}
.layer-controls summary {{ cursor:pointer; list-style:none; padding:.5rem 0; font-weight:750; }}
.layer-controls summary::-webkit-details-marker {{ display:none; }}
.menu-glyph {{ display:inline-block; margin-right:.35rem; }}
.layer-control-body {{ display:flex; flex-wrap:wrap; gap:.55rem 1rem; padding:0 0 .55rem; }}
.layer-canvas {{ position:relative; overflow:hidden; border-radius:14px; }}
.control-group {{ display:flex; align-items:center; flex-wrap:wrap; gap:.35rem .55rem; }}
.control-group > strong {{ margin-right:.15rem; font-size:.78rem; opacity:.75; }}
.base-button {{ border-color:rgba(255,255,255,.18); background:rgba(255,255,255,.06); color:white; padding:.35rem .55rem; }}
.base-button.active {{ border-color:white; background:rgba(255,255,255,.16); font-weight:700; }}

.layer-canvas .layer-base,.layer-svg {{ position:absolute; display:block; width:100%; height:100%; }}
.layer-svg {{ z-index:10; pointer-events:none; }} .layer-svg a {{ pointer-events:auto; }}
.legend {{ display:grid; grid-template-columns:2.2rem 1fr; gap:.55rem .7rem; margin:.8rem 0 1rem; align-items:center; }}
.legend-swatch {{ display:block; width:2rem; box-sizing:border-box; }}
.note {{ color:var(--muted); }} .empty {{ padding:1rem; background:var(--panel-2); border-radius:12px; }} .diagnostic-details {{ margin:.8rem 0; padding:.7rem .85rem; background:var(--panel-2); border:1px solid var(--border); border-radius:12px; }} .diagnostic-details summary {{ cursor:pointer; font-weight:700; }}
.badge {{ display:inline-flex; align-items:center; gap:.35rem; padding:.25rem .55rem; border-radius:999px; background:var(--accent-soft); color:var(--accent); font-size:.76rem; font-weight:700; }}
.metric-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.65rem; margin:.8rem 0 1rem; }}
.metric {{ padding:.75rem .85rem; border:1px solid var(--border); background:var(--panel-2); border-radius:14px; }}
.metric small {{ display:block; color:var(--muted); font-size:.72rem; margin-bottom:.15rem; }} .metric strong {{ font-size:1.15rem; }}
footer {{ margin-top:1.5rem; color:var(--muted); font-size:.8rem; text-align:center; }}
a {{ color:var(--accent); }}
@media (max-width: 900px) {{ .topbar {{ position:static; }} main {{ padding:.8rem; }} .topbar {{ margin:0 -.8rem .8rem; padding:.75rem .8rem; }} th {{ position:static; }} table {{ display:block; overflow:auto; }} }}
</style></head><body>
<main>
<div class="topbar"><div class="brand"><span class="brand-mark"></span><div><div class="brand h1">Prospector</div><div style="font-size:.72rem;color:var(--muted)">{_escape(run_id)}</div></div></div><details class="theme-menu"><summary>Theme</summary><div class="theme-options"><button type="button" data-theme-choice="light">Light</button><button type="button" data-theme-choice="dark">Dark</button><button type="button" data-theme-choice="rose">Rose</button><button type="button" data-theme-choice="lavender">Lavender</button><button type="button" data-theme-choice="mint">Mint</button><button type="button" data-theme-choice="sky">Sky</button><button type="button" data-theme-toggle>Dark / light</button></div></details></div>
<div class="hero"><span class="badge">LiDAR archaeological prospection</span><h2>Landscape evidence at {_escape(location_name or run_id)}</h2><p>Terrain-first discovery with Historic England, modern-map and satellite context.</p></div>
<section><h2>Study area</h2><div class="metric-row"><div class="metric"><small>Location</small><strong>{_escape(location_name or "Coordinate study")}</strong></div><div class="metric"><small>Centre</small><strong>{latitude:.5f}, {longitude:.5f}</strong></div><div class="metric"><small>British National Grid</small><strong>{easting:.0f} E / {northing:.0f} N</strong></div><div class="metric"><small>Diameter</small><strong>{diameter_m:,.0f} m</strong></div><div class="metric"><small>Detections</small><strong>{len(candidates)}</strong></div></div><table><tr><th>Resolved place</th><td>{_escape(location_display_name or location_name or "—")}</td></tr><tr><th>CRS</th><td>EPSG:27700</td></tr><tr><th>Study bounds</th><td>{study_bounds[0]:.2f}, {study_bounds[1]:.2f}, {study_bounds[2]:.2f}, {study_bounds[3]:.2f}</td></tr></table></section>

<section><h2>LiDAR + Historic England Aerial Archaeology Mapping</h2><p class="note">The PNG itself contains the archaeological layer; the interactive map below lets you inspect the same evidence independently.</p>{_he_legend()}{overlay_html}</section>

<section><h2>Interactive evidence map</h2><p class="note">Switch base imagery while keeping the Historic England and Prospector anomaly overlays aligned to the same map grid.</p>{interactive_map_html}</section>

<section id="anomaly-section"><h2>Terrain anomaly scan</h2><p class="note">Hybrid discovery combines multi-scale local relief, persistence, linear/Hough structure, ring/annular response, ridge/valley morphology, local texture/coherence and unsupervised terrain novelty. Historic England remains contextual evidence and an exclusion mask.</p><div class="metric-row"><div class="metric"><small>Sensitivity</small><strong>{_escape(sensitivity_name)}</strong></div><div class="metric"><small>Detector</small><strong>Hybrid + unsupervised ML</strong></div><div class="metric"><small>Detector workers</small><strong>{_escape(detector_workers_label)}</strong></div><div class="metric"><small>Retained candidates</small><strong>{len(candidates)}</strong></div><div class="metric"><small>HE detailed features</small><strong>{len(aim_features)}</strong></div></div>{anomaly_map_html}<table><thead><tr><th>#</th><th>Map</th><th>Class</th><th>Final</th><th>LiDAR</th><th>Persistence</th><th>Morphology</th><th>Linear</th><th>Ring</th><th>Ridge/valley</th><th>Novelty</th><th>Texture</th><th>Modern</th><th>Satellite</th><th>HE similarity</th><th>Relief</th><th>Scale</th><th>Area</th><th>Why retained</th></tr></thead><tbody>{candidate_rows}</tbody></table>{diagnostic_html}<details><summary>Detector parameters and provenance</summary><pre>{metadata_json}</pre></details></section>

<section><h2>Modern feature context</h2><p class="note">Modern mapping is a penalty/context source, not the discovery engine. Roads and buildings are strongly down-ranked; boundaries and tracks are treated more softly.</p><p>Combined modern-context GeoJSON: {_relative_link(destination, modern_context_path)}</p><p>Modernity score raster: {_relative_link(destination, modern_context_raster_path)}</p></section>

<section><h2>Satellite context</h2><p class="note">Sentinel-2 is an optional visual base layer and contextual evidence. It can support a terrain signal through vegetation/reflectance differences but does not define an archaeological candidate.</p>{satellite_html}<table><tr><th>Selected scenes</th><td>{_escape(satellite_meta.get("scene_count", 0))}</td></tr><tr><th>Support raster</th><td>{_relative_link(destination, satellite_support_path)}</td></tr><tr><th>STAC search record</th><td>{_relative_link(destination, satellite_search_path)}</td></tr></table></section>

<section><h2>Combined terrain anomalies + Historic England mapping</h2><p class="note">Review image with anomaly labels and Historic England context rendered on the same LiDAR grid.</p>{combined_map_html}</section>

<section><h2>Historic England Aerial Archaeology Mapping records</h2><p>{len(project_areas)} project area(s), {len(monument_extents)} monument extent(s), and {len(aim_features)} detailed mapped feature geometries.</p><p class="note">Monument extents represent the general recorded monument envelope; detailed mapping represents mapped individual features.</p><table><tr><th>#</th><th>Historic England record</th><th>Type</th><th>Period</th><th>Evidence</th><th>Primary source</th><th>Features</th><th>Details</th></tr>{_aim_record_rows(aim_features)}</table></section>

<section><h2>Run artefacts</h2><p>DTM: {_relative_link(destination, dtm_path)}</p><p>Hillshade: {_relative_link(destination, hillshade_path)}</p><p>AIM GeoJSON: {_relative_link(destination, aim_geojson_path)}</p><p>Candidate GeoJSON: {_relative_link(destination, candidate_geojson_path)}</p><p>LiDAR base: {_relative_link(destination, lidar_base_path)}</p><p>LiDAR anomaly base: {_relative_link(destination, anomaly_base_path)}</p><p>Sentinel-2 base: {_relative_link(destination, satellite_preview_path)}</p><p>AIM visualisation: {_relative_link(destination, overlay_path)}</p><p>Anomaly visualisation: {_relative_link(destination, anomaly_overlay_path)}</p><p>Combined visualisation: {_relative_link(destination, anomaly_aim_overlay_path)}</p></section>

<section><h2>Provenance</h2><p>{cache_hits} cache hit(s), {downloads} download(s).</p><table><tr><th>Status</th><th>URL</th><th>SHA-256</th><th>Bytes</th></tr>{''.join(f'<tr><td>{_escape("HIT" if e.get("cache_hit") else "DOWNLOAD")}</td><td>{_escape(e.get("url", ""))}</td><td><code>{_escape(e.get("sha256", ""))}</code></td><td>{_escape(e.get("size_bytes", ""))}</td></tr>' for e in cache_entries)}</table></section>
<section><h2>Acquisition status</h2>{error_html}</section>
<section><h2>Location attribution</h2><p>{_escape(location_attribution or "No reverse-geocoder attribution supplied.")}</p></section>
{layer_script}<footer>Prospector {_escape(application_version)}. LiDAR elevation supplied by the Environment Agency component used by the NLS Maps 50cm–1m composite; archaeology from Historic England Aerial Investigation and Mapping; modern context from Ordnance Survey/OpenStreetMap; satellite context from Sentinel-2 via Microsoft Planetary Computer.</footer>
</main></body></html>"""
    destination.write_text(document, encoding="utf-8")
    return destination
