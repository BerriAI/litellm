FROM rust:1.87-slim-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir "headroom-ai[proxy]==0.27.0"


FROM python:3.12-slim-bookworm

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8787

ENV HEADROOM_TELEMETRY=off

CMD ["headroom", "proxy", "--host", "0.0.0.0", "--port", "8787", "--no-optimize"]
