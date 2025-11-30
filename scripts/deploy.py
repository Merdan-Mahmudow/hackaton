#!/usr/bin/env python3
"""Deployment script for ML-Web application.

This script prepares the application for deployment by:
- Applying database migrations
- Verifying database connectivity
- Checking required directories and files
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], cwd: Optional[Path] = None) -> bool:
    """Run a shell command and return True if successful."""
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=cwd or project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        return False


def check_directories() -> bool:
    """Check that required directories exist."""
    logger.info("Checking required directories...")
    required_dirs = [
        project_root / "data",
        project_root / "models",
        project_root / "reports",
        project_root / "backend",
        project_root / "frontend",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                all_exist = False
        else:
            logger.debug(f"Directory exists: {dir_path}")
    
    return all_exist


def apply_migrations() -> bool:
    """Apply Alembic database migrations."""
    logger.info("Applying database migrations...")
    
    # Check if alembic is available
    try:
        import alembic
    except ImportError:
        logger.error("Alembic is not installed. Please install dependencies first.")
        logger.info("Run: uv sync")
        return False
    
    # Try to use uv run if available, otherwise use direct command
    use_uv = False
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        use_uv = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.debug("uv not available, using direct commands")
    
    alembic_cmd = ["uv", "run", "alembic"] if use_uv else ["alembic"]
    
    # Check current migration version
    try:
        result = subprocess.run(
            alembic_cmd + ["current"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if "af27c5d5ed93" in result.stdout or "head" in result.stdout.lower():
            logger.info("Database is already at the latest migration version")
            return True
    except Exception as e:
        logger.debug(f"Could not check current version: {e}")
    
    # Check if tables already exist (manual migration scenario)
    try:
        from backend.app.database import engine
        from sqlalchemy import inspect
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        required_tables = ["feedback_entries", "prediction_records"]
        
        if all(t in tables for t in required_tables):
            logger.info("Tables already exist, stamping database with current migration")
            # Stamp the database with the current migration version
            if run_command(alembic_cmd + ["stamp", "head"]):
                logger.info("Database stamped successfully")
                return True
    except Exception as e:
        logger.debug(f"Could not check tables: {e}")
    
    # Run alembic upgrade head
    return run_command(alembic_cmd + ["upgrade", "head"])


def verify_database() -> bool:
    """Verify database connectivity and schema."""
    logger.info("Verifying database...")
    
    try:
        from backend.app.database import engine, Base
        from backend.app.database import FeedbackEntry, PredictionRecord
        
        # Try to connect to database
        with engine.connect() as conn:
            logger.info("Database connection successful")
        
        # Check if tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        required_tables = ["feedback_entries", "prediction_records"]
        missing_tables = [t for t in required_tables if t not in tables]
        
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            logger.info("Creating tables...")
            Base.metadata.create_all(bind=engine)
            logger.info("Tables created successfully")
        else:
            logger.info(f"All required tables exist: {required_tables}")
        
        return True
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False


def main() -> int:
    """Main deployment function."""
    logger.info("Starting deployment process...")
    logger.info(f"Project root: {project_root}")
    
    # Step 1: Check directories
    if not check_directories():
        logger.error("Directory check failed")
        return 1
    
    # Step 2: Apply migrations
    if not apply_migrations():
        logger.error("Migration application failed")
        return 1
    
    # Step 3: Verify database
    if not verify_database():
        logger.error("Database verification failed")
        return 1
    
    logger.info("Deployment completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

