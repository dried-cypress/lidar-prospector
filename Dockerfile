FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE /workspace/
COPY src /workspace/src
COPY tests /workspace/tests
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e '.[all]'

CMD ["sleep", "infinity"]
