FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY server.py .
COPY writing_kb/ writing_kb/
COPY craft/ craft/
COPY style/ style/
COPY structure/ structure/
EXPOSE 10000
CMD ["uv", "run", "python", "server.py"]
