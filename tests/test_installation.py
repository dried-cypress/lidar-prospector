from pathlib import Path


def test_install_script_forces_project_venv_and_no_user_install() -> None:
    root = Path(__file__).parents[1]
    script = (root / "install.sh").read_text(encoding="utf-8")
    assert 'VENV_DIR="${ROOT_DIR}/.venv"' in script
    assert '"${VENV_PYTHON}" -m pip install --no-user' in script
    assert 'VENV_PROSPECTOR="${VENV_DIR}/bin/prospector"' in script


def test_module_entry_point_exists() -> None:
    root = Path(__file__).parents[1]
    text = (root / "src/prospector/__main__.py").read_text(encoding="utf-8")
    assert 'from prospector.cli import app' in text
