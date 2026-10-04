from __future__ import annotations

import sys


def test_runtime_supports_python_310_plus() -> None:
    assert sys.version_info >= (3, 10)


def test_no_python_311_datetime_utc_imports_in_runtime() -> None:
    from pathlib import Path

    source_root = Path(__file__).parents[1] / "src" / "prospector"
    offenders = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from datetime import UTC" in text:
            offenders.append(str(path))
    assert not offenders


def test_cli_project_venv_guard_rejects_missing_virtual_env(monkeypatch) -> None:
    from prospector.cli import _require_project_venv

    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    try:
        _require_project_venv()
    except RuntimeError as exc:
        assert "virtual environment" in str(exc)
    else:
        raise AssertionError("Expected Prospector to reject execution outside a virtual environment")
