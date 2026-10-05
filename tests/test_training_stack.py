from pathlib import Path


def test_compose_defines_postgis_health_and_training_services() -> None:
    root = Path(__file__).parents[1]
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    assert "postgis/postgis:17-3.5" in compose
    assert "condition: service_healthy" in compose
    assert "trainer:" in compose
    assert "PROSPECTOR_DATABASE_URL" in compose


def test_gitignore_keeps_project_and_environment_protection() -> None:
    root = Path(__file__).parents[1]
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert "project/" in text
    assert ".env" in text
    assert "env/" in text


def test_all_extra_expands_geospatial_and_training_dependencies() -> None:
    root = Path(__file__).parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "all = [" in text
    for requirement in ("numpy>=2.2,<3", "rasterio>=1.4,<2", "scipy>=1.15,<2", "scikit-image>=0.25,<1", "joblib>=1.5,<2", "psycopg[binary]>=3.2,<4"):
        assert requirement in text
    assert 'all = ["prospector[geo,dev,training]"]' not in text


def test_dockerfile_installs_geospatial_and_training_extras_and_verifies_them() -> None:
    root = Path(__file__).parents[1]
    text = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "-e '.[geo,training]'" in text
    assert "--no-user" in text
    assert "import fiona, joblib, matplotlib, numpy, PIL, psycopg" in text


def test_training_cheatsheet_is_present_and_contains_core_commands() -> None:
    root = Path(__file__).parents[1]
    text = (root / "notes.txt").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("Prospector training stack:")
    for command in (
        "docker compose up -d",
        "prospector train init",
        "prospector train ingest-he",
        "prospector train build-dataset",
        "prospector train fit",
        "prospector train status",
    ):
        assert command in text


def test_readme_documents_docker_geospatial_rebuild() -> None:
    root = Path(__file__).parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    assert "docker compose build --no-cache app trainer" in text
    assert "V0.5.1" in text
