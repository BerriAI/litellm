ARG BASE_IMAGE=docker.litellm.ai/berriai/litellm
ARG BASE_TAG=main-stable
FROM ${BASE_IMAGE}:${BASE_TAG}

USER root

# Install New Relic agent (ensurepip bootstraps pip in case base image venv omits it)
RUN python -m ensurepip && python -m pip install --no-cache-dir 'newrelic>=12.1.0,<13'

# Copy New Relic-specific configuration files
COPY docker/newrelic/supervisord.conf /etc/supervisord_newrelic.conf
COPY docker/newrelic/entrypoint.sh /app/docker/newrelic/entrypoint.sh
RUN chmod +x /app/docker/newrelic/entrypoint.sh

# Override entrypoint to always use newrelic-admin
ENTRYPOINT ["/app/docker/newrelic/entrypoint.sh"]

LABEL org.opencontainers.image.description="LiteLLM with New Relic APM and AI monitoring"
LABEL org.opencontainers.image.documentation="https://docs.litellm.ai/docs/proxy/observability/new_relic"
