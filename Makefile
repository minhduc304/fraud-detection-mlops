.PHONY: lint test train serve up

lint:
	uv run ruff check src/ && uv run mypy src/

test:
	uv run pytest tests/unit/ -q

train:
	uv run python -m fraudstream.training.train

serve:
	@echo "Phase 4 not implemented yet"

up:
	docker compose up
