#!/bin/bash
# Deployment script for ML-Web application

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting deployment process..."
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv to run deployment script..."
    uv run python scripts/deploy.py
else
    echo "Error: uv is required. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Deployment completed!"

