from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS aim_features (
    id BIGSERIAL PRIMARY KEY,
    source_layer TEXT NOT NULL,
    source_uid TEXT NOT NULL,
    monument_type TEXT,
    period TEXT,
    evidence TEXT,
    project TEXT,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    geom geometry(Geometry, 27700) NOT NULL,
    geometry_hash TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_layer, source_uid, geometry_hash)
);
CREATE INDEX IF NOT EXISTS aim_features_geom_gix ON aim_features USING GIST (geom);
CREATE INDEX IF NOT EXISTS aim_features_layer_idx ON aim_features (source_layer);
CREATE INDEX IF NOT EXISTS aim_features_uid_idx ON aim_features (source_uid);

CREATE TABLE IF NOT EXISTS training_examples (
    id BIGSERIAL PRIMARY KEY,
    source_feature_id BIGINT REFERENCES aim_features(id) ON DELETE SET NULL,
    label SMALLINT NOT NULL CHECK (label IN (0,1)),
    example_type TEXT NOT NULL,
    scale_m DOUBLE PRECISION NOT NULL,
    rotation_deg DOUBLE PRECISION NOT NULL,
    sample_key TEXT NOT NULL,
    group_key TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (sample_key)
);
CREATE INDEX IF NOT EXISTS training_examples_label_idx ON training_examples (label);
CREATE INDEX IF NOT EXISTS training_examples_group_idx ON training_examples (group_key);

CREATE TABLE IF NOT EXISTS model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    model_path TEXT NOT NULL,
    dataset_path TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    feature_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'trained',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_name, version)
);
CREATE INDEX IF NOT EXISTS model_registry_status_idx ON model_registry (status);
"""


def database_url_from_env() -> str:
    value = os.getenv("PROSPECTOR_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError(
            "No training database configured. Set PROSPECTOR_DATABASE_URL, "
            "for example postgresql://prospector:prospector@localhost:5432/prospector."
        )
    return value


def _psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Training database support requires psycopg. Install with `pip install -e '.[training]'`."
        ) from exc
    return psycopg


def connect(database_url: str | None = None) -> Any:
    psycopg = _psycopg()
    return psycopg.connect(database_url or database_url_from_env())


def ensure_schema(database_url: str | None = None) -> None:
    with connect(database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
        conn.commit()


def _property(properties: dict[str, Any], *names: str) -> str:
    for name in names:
        value = properties.get(name)
        if value not in (None, "", "-"):
            return str(value)
    return ""


def feature_uid(feature: dict[str, Any], fallback: int) -> str:
    properties = feature.get("properties") or {}
    return _property(properties, "HE_UID", "DHEUID_1", "HER_NO", "HERNO_1", "OBJECTID", "FID", "id") or f"feature-{fallback}"


def geometry_hash(feature: dict[str, Any]) -> str:
    from shapely.geometry import shape
    geometry = shape(feature["geometry"])
    return hashlib.sha256(geometry.wkb).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredFeature:
    id: int
    source_layer: str
    source_uid: str
    monument_type: str
    period: str
    evidence: str
    project: str


def upsert_features(features: Iterable[dict[str, Any]], database_url: str | None = None) -> int:
    psycopg = _psycopg()
    from psycopg.types.json import Jsonb

    rows = list(features)
    if not rows:
        return 0
    count = 0
    with connect(database_url) as conn:
        with conn.cursor() as cursor:
            for index, feature in enumerate(rows, 1):
                properties = dict(feature.get("properties") or {})
                source_layer = _property(properties, "prospector_aim_layer", "layer") or "Unknown"
                uid = feature_uid(feature, index)
                monument_type = _property(properties, "MONUMENT_TYPE")
                period = _property(properties, "PERIOD")
                evidence = _property(properties, "EVIDENCE_1")
                project = _property(properties, "PROJECT", "project")
                geometry_wkt = __import__("shapely.geometry", fromlist=["shape"]).shape(feature["geometry"]).wkt
                ghash = geometry_hash(feature)
                cursor.execute(
                    """
                    INSERT INTO aim_features
                      (source_layer, source_uid, monument_type, period, evidence, project, properties, geom, geometry_hash)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,ST_Force2D(ST_SetSRID(ST_GeomFromText(%s),27700)),%s)
                    ON CONFLICT (source_layer, source_uid, geometry_hash)
                    DO UPDATE SET properties=EXCLUDED.properties,
                                  monument_type=EXCLUDED.monument_type,
                                  period=EXCLUDED.period,
                                  evidence=EXCLUDED.evidence,
                                  project=EXCLUDED.project
                    RETURNING id
                    """,
                    (source_layer, uid, monument_type, period, evidence, project, Jsonb(properties), geometry_wkt, ghash),
                )
                if cursor.fetchone():
                    count += 1
        conn.commit()
    return count


def query_features(
    bbox: tuple[float, float, float, float],
    *,
    layer: str = "Detailed_Mapping",
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    xmin, ymin, xmax, ymax = bbox
    with connect(database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, source_layer, source_uid, monument_type, period, evidence, project,
                       ST_AsGeoJSON(geom), properties
                FROM aim_features
                WHERE source_layer = %s
                  AND geom && ST_MakeEnvelope(%s,%s,%s,%s,27700)
                  AND ST_Intersects(geom, ST_MakeEnvelope(%s,%s,%s,%s,27700))
                ORDER BY id
                """,
                (layer, xmin, ymin, xmax, ymax, xmin, ymin, xmax, ymax),
            )
            rows = cursor.fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append({
            "id": int(row[0]),
            "source_layer": row[1],
            "source_uid": row[2],
            "monument_type": row[3] or "",
            "period": row[4] or "",
            "evidence": row[5] or "",
            "project": row[6] or "",
            "geometry": json.loads(row[7]),
            "properties": row[8] or {},
        })
    return result


def register_model(
    *,
    model_name: str,
    version: str,
    model_path: str,
    dataset_path: str,
    metrics: dict[str, Any],
    feature_schema: dict[str, Any],
    database_url: str | None = None,
) -> None:
    from psycopg.types.json import Jsonb
    with connect(database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_registry(model_name,version,model_path,dataset_path,metrics,feature_schema,status)
                VALUES (%s,%s,%s,%s,%s,%s,'trained')
                ON CONFLICT (model_name,version) DO UPDATE SET
                  model_path=EXCLUDED.model_path,
                  dataset_path=EXCLUDED.dataset_path,
                  metrics=EXCLUDED.metrics,
                  feature_schema=EXCLUDED.feature_schema,
                  status='trained'
                """,
                (model_name, version, model_path, dataset_path, Jsonb(metrics), Jsonb(feature_schema)),
            )
        conn.commit()


def register_training_manifest(
    manifest_rows: Iterable[dict[str, Any]],
    database_url: str | None = None,
) -> int:
    rows = list(manifest_rows)
    if not rows:
        return 0
    with connect(database_url) as conn:
        with conn.cursor() as cursor:
            count = 0
            for row in rows:
                source_uid = str(row.get("source_uid") or "")
                source_feature_id = None
                if source_uid:
                    cursor.execute(
                        "SELECT id FROM aim_features WHERE source_uid=%s AND source_layer='Detailed_Mapping' ORDER BY id LIMIT 1",
                        (source_uid,),
                    )
                    result = cursor.fetchone()
                    source_feature_id = int(result[0]) if result else None
                sample_key = str(row.get("patch_path") or row.get("tile") or "") + ":" + str(row.get("scale_m", "")) + ":" + str(row.get("rotation_deg", "")) + ":" + str(row.get("index", ""))
                cursor.execute(
                    """
                    INSERT INTO training_examples
                      (source_feature_id,label,example_type,scale_m,rotation_deg,sample_key,group_key,metadata)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (sample_key) DO NOTHING
                    """,
                    (
                        source_feature_id, int(row.get("label", 0)), str(row.get("example_type", "background")),
                        float(row.get("scale_m", 0.0)), float(row.get("rotation_deg", 0.0)),
                        sample_key, str(row.get("source_uid") or row.get("tile") or "unknown"), json.dumps(row),
                    ),
                )
                count += 1
        conn.commit()
    return count
