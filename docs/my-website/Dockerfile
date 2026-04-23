ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10.9

FROM $UV_IMAGE AS uvbin

FROM python:3.14.0a3-slim

COPY --from=uvbin /uv /usr/local/bin/uv
COPY --from=uvbin /uvx /usr/local/bin/uvx
COPY . /app
WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python python

EXPOSE $PORT 

CMD ["sh", "-c", "litellm --host 0.0.0.0 --port $PORT --workers 10 --config config.yaml"]
