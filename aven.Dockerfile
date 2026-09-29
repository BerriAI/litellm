ARG LITELLM_BASE_IMAGE=litellm-base:local
FROM ${LITELLM_BASE_IMAGE}

ARG MCP_HOME=/app/mcp

COPY --chown=65534:0 .mcp-build/ ${MCP_HOME}/
COPY --chown=65534:0 config/mcp-servers.yml ${MCP_HOME}/mcp-servers.yml
COPY --chown=65534:0 docker/merge_and_start.py /app/docker/merge_and_start.py

USER 65534

ENTRYPOINT ["python", "/app/docker/merge_and_start.py"]