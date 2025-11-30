#!/bin/bash
# Docker entrypoint script for ML-Web application

set -e

echo "Starting ML-Web application..."

# Check if tables exist before applying migrations
echo "Checking database state..."
uv run python -c "
from backend.app.database import engine
from sqlalchemy import inspect

inspector = inspect(engine)
tables = inspector.get_table_names()
required_tables = ['feedback_entries', 'prediction_records']

if all(t in tables for t in required_tables):
    print('Tables already exist, stamping database...')
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'alembic', 'stamp', 'head'], check=False)
else:
    print('Tables do not exist, will apply migrations...')
" || true

# Apply database migrations
echo "Applying database migrations..."
uv run alembic upgrade head || {
    echo "Migration failed, attempting to stamp database..."
    uv run alembic stamp head || true
}

# Verify database
echo "Verifying database..."
uv run python -c "
from backend.app.database import engine, Base
from backend.app.database import FeedbackEntry, PredictionRecord
from sqlalchemy import inspect

inspector = inspect(engine)
tables = inspector.get_table_names()
required_tables = ['feedback_entries', 'prediction_records']

missing_tables = [t for t in required_tables if t not in tables]
if missing_tables:
    print(f'Creating missing tables: {missing_tables}')
    Base.metadata.create_all(bind=engine)
    print('Tables created successfully')
else:
    print(f'All required tables exist: {required_tables}')
"

# Start the application
echo "Starting uvicorn server..."
exec "$@"

