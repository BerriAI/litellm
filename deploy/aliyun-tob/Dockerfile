FROM docker.litellm.ai/berriai/litellm:main-stable@sha256:a1745e629abfb17d434426ff48b115f54f4f4c4a0f5af241de569e93c63c411e

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PRISMA_BINARY_CACHE_DIR=/opt/prisma/binaries \
    PRISMA_CLI_PATH=/opt/prisma/binaries/node_modules/.bin/prisma \
    PRISMA_CLI_QUERY_ENGINE_TYPE=binary \
    PRISMA_OFFLINE_MODE=true

RUN /app/.venv/bin/python -m ensurepip && \
    /app/.venv/bin/python -m pip install --no-cache-dir argon2-cffi==25.1.0

COPY litellm /app/.venv/lib/python3.13/site-packages/litellm
COPY litellm-proxy-extras/litellm_proxy_extras/schema.prisma /app/.venv/lib/python3.13/site-packages/litellm_proxy_extras/schema.prisma
COPY litellm-proxy-extras/litellm_proxy_extras/migrations/20260724000000_add_public_relay_v1 /app/.venv/lib/python3.13/site-packages/litellm_proxy_extras/migrations/20260724000000_add_public_relay_v1
COPY ui/litellm-dashboard/out /app/.venv/lib/python3.13/site-packages/litellm/proxy/_experimental/out
COPY schema.prisma /app/schema.prisma
COPY litellm-proxy-extras /app/litellm-proxy-extras

EXPOSE 4000/tcp

ENTRYPOINT ["/app/docker/prod_entrypoint.sh"]
CMD ["--port", "4000"]
