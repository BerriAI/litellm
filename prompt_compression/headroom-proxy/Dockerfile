FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir "headroom-ai[proxy]==0.27.0"

RUN python -c "from headroom.proxy.server import run_server; print('import ok')"

EXPOSE 8787

ENV HEADROOM_TELEMETRY=off

CMD ["headroom", "proxy", "--host", "0.0.0.0", "--port", "8787", "--no-optimize"]
