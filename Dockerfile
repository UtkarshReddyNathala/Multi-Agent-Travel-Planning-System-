FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# The AviationStack MCP server is started at runtime with `uvx aviationstack-mcp`
# (see mcp_client.py). `uv` provides both `uv` and `uvx`. That package requires
# Python >= 3.13, so uvx downloads its own interpreter and environment the first
# time it runs.
RUN pip install --no-cache-dir uv==0.12.17

# Do not run as root.
RUN useradd --create-home --uid 10001 app
ENV UV_CACHE_DIR=/home/app/.cache/uv
ENV UV_PYTHON_INSTALL_DIR=/home/app/.local/share/uv/python

COPY --chown=app:app . .

USER app

# Warm uvx's cache at build time so the first flight request does not have to
# download Python 3.13 inside the MCP timeout. Not fatal: uvx would simply do the
# same download on first use.
RUN uvx --from aviationstack-mcp python -c "import mcp" \
    || echo "WARNING: could not pre-warm aviationstack-mcp; it will be fetched on first use"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
