FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir "headroom-ai[proxy]==0.27.0" \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 10000

ENV HEADROOM_TELEMETRY=off

CMD ["headroom", "proxy", "--host", "0.0.0.0", "--port", "10000"]
