.PHONY: lint test train serve up

lint:
	uv run ruff check src/ && uv run mypy src/

test:
	uv run pytest tests/unit/ -q

train:
	@echo "Phase 1 not implemented yet"

serve:
	@echo "Phase 4 not implemented yet"

up:
	docker compose up
