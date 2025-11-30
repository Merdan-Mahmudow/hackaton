FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Copy application code
COPY backend ./backend
COPY frontend ./frontend
COPY ml ./ml
COPY data ./data
COPY models ./models

# Copy Alembic configuration and migrations
COPY alembic.ini ./
COPY alembic ./alembic

# Copy deployment scripts
COPY scripts ./scripts
RUN chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000

# Use entrypoint script to run migrations before starting the app
ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
