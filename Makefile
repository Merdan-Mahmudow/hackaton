UV ?= uv
PYTHON ?= python
UVICORN ?= uvicorn

.PHONY: install install-uv train train-transformer eda evaluate serve docker-build docker-up docker-down feedback-export history-report test deploy

install:
	@echo "Installing dependencies with uv..."
	$(UV) sync

install-uv:
	@echo "Installing uv..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh

deploy:
	@echo "Running deployment script..."
	$(UV) run python scripts/deploy.py

train:
	$(UV) run python ml/train_baseline.py

train-transformer:
	$(UV) run python ml/train_transformer.py

eda:
	$(UV) run python ml/eda.py

evaluate:
	$(UV) run python ml/evaluate_model.py

feedback-export:
	$(UV) run python ml/feedback_to_dataset.py

history-report:
	$(UV) run python ml/history_report.py

test:
	$(UV) run python -m unittest discover -s tests -p 'test_*.py'

serve:
	$(UV) run uvicorn backend.app.main:app --reload

docker-build:
docker compose build

docker-up:
docker compose up --build

docker-down:
docker compose down
