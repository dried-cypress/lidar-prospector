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


## Docker training stack

V0.5 adds an optional PostGIS-backed training stack. Docker Compose starts PostgreSQL/PostGIS plus separate application and trainer containers; Compose waits for the database health check before starting the dependent services.

```bash
cp .env.example .env
docker compose up -d
docker compose run --rm trainer prospector train init
```

Use `prospector train ingest-he` to ingest HE AIM in bounded spatial chunks, `prospector train build-dataset --download-lidar` to create multi-scale/rotation-augmented LiDAR examples, and `prospector train fit` to train and register the local model.
