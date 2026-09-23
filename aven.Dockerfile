ARG LITELLM_BASE_IMAGE=litellm-base:local
FROM ${LITELLM_BASE_IMAGE}

USER root

RUN mkdir -p /opt/mcp /config

COPY .mcp-build/ /opt/mcp/

COPY config/mcp-servers.yml /opt/mcp/mcp-servers.yml

RUN chown -R 65534:0 /opt/mcp /config && \
    chmod -R a+rX /opt/mcp && \
    chmod 750 /config

USER 65534
