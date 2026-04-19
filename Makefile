.PHONY: install test dev docker-build docker-run docker-stop clean

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

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
