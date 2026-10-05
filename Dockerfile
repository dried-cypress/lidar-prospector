FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_USER=0

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE /workspace/
COPY src /workspace/src
COPY tests /workspace/tests
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --no-user -e '.[geo,training]' \
    && python - <<'PY2'
import fiona, joblib, matplotlib, numpy, PIL, psycopg, pyproj, rasterio, scipy, shapely, skimage, sklearn
print('Prospector Docker geospatial/training dependencies verified')
PY2

CMD ["sleep", "infinity"]
