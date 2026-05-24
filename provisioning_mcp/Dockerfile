# LiteLLM provisioning MCP server.
#
# Build from the repository ROOT so the helm/litellm chart is in context:
#   docker build -f provisioning_mcp/Dockerfile -t litellm-provisioning-mcp .

# ---- tools: fetch + checksum-verify pinned helm and kubectl ----
FROM debian:bookworm-slim AS tools
ARG HELM_VERSION=3.16.3
ARG KUBECTL_VERSION=1.31.3
# Pinned digest of the linux/amd64 kubectl binary (dl.k8s.io .../kubectl.sha256).
ARG KUBECTL_SHA256=981f6b49577068bc174275184d8ee7105d8e54f40733792c519cd85023984c0f

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tar \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/tools

# helm: download the release tarball and its official sha256sum sidecar, then
# verify before extracting (no piping a remote script into a shell).
RUN curl -fsSLO "https://get.helm.sh/helm-v${HELM_VERSION}-linux-amd64.tar.gz" \
 && curl -fsSLO "https://get.helm.sh/helm-v${HELM_VERSION}-linux-amd64.tar.gz.sha256sum" \
 && EXPECTED="$(awk '{print $1}' "helm-v${HELM_VERSION}-linux-amd64.tar.gz.sha256sum")" \
 && echo "${EXPECTED}  helm-v${HELM_VERSION}-linux-amd64.tar.gz" | sha256sum -c - \
 && tar -xzf "helm-v${HELM_VERSION}-linux-amd64.tar.gz" \
 && install -m 0755 linux-amd64/helm /usr/local/bin/helm

# kubectl: download the binary and verify against the hardcoded digest.
RUN curl -fsSLo kubectl "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
 && echo "${KUBECTL_SHA256}  kubectl" | sha256sum -c - \
 && install -m 0755 kubectl /usr/local/bin/kubectl

# ---- runtime ----
FROM python:3.11-slim AS runtime

COPY --from=tools /usr/local/bin/helm /usr/local/bin/helm
COPY --from=tools /usr/local/bin/kubectl /usr/local/bin/kubectl

WORKDIR /app
COPY provisioning_mcp/pyproject.toml /app/provisioning_mcp/pyproject.toml
COPY provisioning_mcp/src /app/provisioning_mcp/src
RUN pip install --no-cache-dir /app/provisioning_mcp

# The chart this server provisions, baked into the image.
COPY helm/litellm /app/helm/litellm
ENV HELM_CHART_PATH=/app/helm/litellm

RUN useradd --create-home --uid 10001 mcp && chown -R mcp:mcp /app
USER mcp

ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8080
EXPOSE 8080

ENTRYPOINT ["litellm-provisioning-mcp"]
