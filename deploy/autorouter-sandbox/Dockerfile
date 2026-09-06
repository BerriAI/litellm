ARG BASE_IMAGE=ghcr.io/berriai/litellm:main-latest
FROM ${BASE_IMAGE}

ARG SITE_PACKAGES=/app/.venv/lib/python3.13/site-packages
COPY litellm/ ${SITE_PACKAGES}/litellm/
COPY enterprise/litellm_enterprise/ ${SITE_PACKAGES}/litellm_enterprise/
COPY enterprise/ /app/enterprise/
COPY litellm-proxy-extras/litellm_proxy_extras/ ${SITE_PACKAGES}/litellm_proxy_extras/
COPY litellm-proxy-extras/ /app/litellm-proxy-extras/
COPY schema.prisma /app/schema.prisma
COPY deploy/autorouter-sandbox/config.yaml /app/config.yaml

CMD ["--config", "/app/config.yaml", "--port", "4000"]
