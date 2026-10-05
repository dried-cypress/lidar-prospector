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
