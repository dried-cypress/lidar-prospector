# Installation

Prospector must be installed into the repository-local `.venv`. Do not rely on a system `python3` or user-site pip configuration.

The supported installation command is:

```bash
./install.sh
```

The script invokes `.venv/bin/python -m pip ... --no-user`, guaranteeing that the console entry point is created at `.venv/bin/prospector`.

For an existing environment:

```bash
.venv/bin/python -m pip install --no-user -e '.[all]'
```

If `prospector` resolves to `~/.local/bin/prospector`, that is an older user-site installation, not the project-local executable. The project-local executable is `.venv/bin/prospector`.
