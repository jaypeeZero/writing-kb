.PHONY: install test dev docker-build docker-run docker-stop up down restart logs clean

install:
	uv sync

test:
	uv run pytest

dev:
	uv run python server.py

docker-build:
	docker build -t writing-kb .

docker-run:
	docker run -d --name writing-kb --network host writing-kb

docker-stop:
	docker stop writing-kb && docker rm writing-kb

# Compose targets — the supported way to run the HTTP container (port 8880)
up:
	docker compose up -d --build

down:
	docker compose down

# Pick up KB edits in kb_search (index is built at startup)
restart:
	docker compose restart

logs:
	docker compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
