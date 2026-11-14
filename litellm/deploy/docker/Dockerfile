# syntax=docker/dockerfile:1.7

# Base image for building
ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:ae9575b121f5776e2f04e192064bc2c5665af9d1c425d620dc773aca31698712

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:ae9575b121f5776e2f04e192064bc2c5665af9d1c425d620dc773aca31698712
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a
# Pinned by digest like the other base images; bump explicitly on Node upgrades.
ARG UI_BUILD_IMAGE=node:24.19-alpine3.24@sha256:d32cdf619f63fe0471182d08996dd516c6275bb5fd31ae06e55a570bd9e1ad43
# Checksum from https://www.pgbouncer.org/downloads/ (the Wolfi repo only carries 1.24.x)
ARG PGBOUNCER_VERSION=1.26.0
ARG PGBOUNCER_SHA256=afd25dd61ee6775d37b40629b87ce08736b3e6955f3057bb212e410fbf21c71d

FROM $UV_IMAGE AS uvbin

FROM $LITELLM_BUILD_IMAGE AS pgbouncer-builder
ARG PGBOUNCER_VERSION
ARG PGBOUNCER_SHA256
USER root
RUN apk add --no-cache build-base pkgconf libevent-dev openssl-dev curl
WORKDIR /build
RUN curl -fsSL -o pgbouncer.tar.gz "https://www.pgbouncer.org/downloads/files/${PGBOUNCER_VERSION}/pgbouncer-${PGBOUNCER_VERSION}.tar.gz" && \
    echo "${PGBOUNCER_SHA256}  pgbouncer.tar.gz" | sha256sum -c - && \
    tar xzf pgbouncer.tar.gz --strip-components=1 && \
    ./configure --prefix=/usr/local --with-openssl=/usr && \
    make -j"$(nproc)" pgbouncer && \
    install -m 0755 pgbouncer /usr/local/bin/pgbouncer

# Admin UI builder. Pinned to the build platform so the architecture-independent
# Next.js static export compiles once natively even in a multi-arch build,
# instead of once per target arch under QEMU.
FROM --platform=$BUILDPLATFORM $UI_BUILD_IMAGE AS ui-builder

ENV NEXT_TELEMETRY_DISABLED=1 \
    npm_config_fund=false \
    npm_config_audit=false

WORKDIR /ui

COPY litellm/ui/litellm-dashboard/package.json litellm/ui/litellm-dashboard/package-lock.json ./
COPY litellm/ui/litellm-dashboard/vendor/ ./vendor/
RUN --mount=type=cache,target=/root/.npm npm ci --prefer-offline

COPY litellm/ui/litellm-dashboard/ ./
RUN npm run build

# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

WORKDIR /app
USER root

COPY --from=uvbin /uv /usr/local/bin/uv
COPY --from=uvbin /uvx /usr/local/bin/uvx

RUN apk add --no-cache \
    bash \
    gcc \
    python-3.13 \
    python-3.13-dev \
    rust \
    openssl \
    libssl3 \
    openssl-dev \
    nodejs \
    npm \
    libsndfile

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    CARGO_NET_RETRY=5 \
    CARGO_HTTP_MULTIPLEXING=false \
    SQLX_OFFLINE=true \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:${PATH}"

# Copy dependency metadata first for layer caching
COPY pyproject.toml uv.lock ./
COPY litellm/enterprise/pyproject.toml litellm/enterprise/
COPY litellm/proxy-extras/pyproject.toml litellm/proxy-extras/
COPY litellm/packaging/litellm-proxy/pyproject.toml litellm/packaging/litellm-proxy/

# Install third-party dependencies (cached unless pyproject.toml/uv.lock change)
RUN uv sync --frozen --no-install-project --no-install-workspace --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --group admin-mcp \
    --extra extra_proxy \
    --extra semantic-router \
    --extra saml \
    --extra bedrock-realtime \
    --python python3.13

# Copy full source tree
COPY . .

# Replace the committed UI bundle with the one built from this exact source.
# Clearing first drops the committed bundle's content-hashed chunks that COPY
# would otherwise leave behind alongside the fresh ones.
RUN rm -rf litellm/proxy/_experimental/out
COPY --from=ui-builder /ui/out/. litellm/proxy/_experimental/out/

# Build Admin UI before final sync (applies the enterprise color override when present)
RUN sed -i 's/\r$//' litellm/deploy/docker/build_admin_ui.sh && chmod +x litellm/deploy/docker/build_admin_ui.sh && ./litellm/deploy/docker/build_admin_ui.sh

# Install project and workspace packages (fast - deps already cached)
ARG MATURIN_PEP517_ARGS=""
RUN uv build --package litellm-proxy --wheel --out-dir /wheels
RUN uv sync --frozen --no-install-package litellm-proxy --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --group admin-mcp \
    --extra extra_proxy \
    --extra semantic-router \
    --extra saml \
    --extra bedrock-realtime \
    --python python3.13

RUN uv pip install --python /app/.venv/bin/python --no-deps /wheels/litellm_proxy-*.whl

RUN cp -r litellm/enterprise litellm/runtime "$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/litellm/"

RUN HOME=/opt/prisma XDG_CACHE_HOME=/opt/prisma/.cache PRISMA_BINARY_CACHE_DIR=/opt/prisma/binaries \
    npm_config_cache=/root/.npm \
    prisma generate --schema=litellm/schema.prisma

RUN sed -i 's/\r$//' litellm/deploy/docker/docker-entrypoint.sh && chmod +x litellm/deploy/docker/docker-entrypoint.sh

RUN mkdir -p /var/lib/litellm/ui /var/lib/litellm/assets && \
    cp -r litellm/proxy/_experimental/out/. /var/lib/litellm/ui/ && \
    cp litellm/proxy/logo.png /var/lib/litellm/assets/logo.png && \
    touch /var/lib/litellm/ui/.litellm_ui_ready

FROM $LITELLM_BUILD_IMAGE AS liteadmin-builder
COPY --from=uvbin /uv /usr/local/bin/uv
RUN apk add --no-cache python-3.13
ADD --checksum=sha256:4fb1847e28e2be30ba76dac94f02c4903291326516a0f4ba772f6ffb8b25d05c \
    https://github.com/BerriAI/litellm-admin-agent/archive/5661c805ce479a0cf11b842ea133cf8d22cd7e34.tar.gz /tmp/liteadmin.tar.gz
RUN mkdir /tmp/liteadmin && tar xzf /tmp/liteadmin.tar.gz --strip-components=1 -C /tmp/liteadmin && \
    uv venv /opt/liteadmin --python python3.13 && \
    uv pip install --python /opt/liteadmin/bin/python --require-hashes -r /tmp/liteadmin/requirements.txt && \
    uv pip install --python /opt/liteadmin/bin/python --no-deps /tmp/liteadmin

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime
ARG LITELLM_RELEASE_TAG=""
ENV LITELLM_RELEASE_TAG=${LITELLM_RELEASE_TAG}

USER root

# The base image only configures Chainguard's authenticated apk repo, which
# requires an enterprise subscription. Add the public Wolfi repo so `apk add`
# also works for anyone installing extra packages into a running container.
# https://github.com/BerriAI/litellm/issues/33518
RUN echo "https://packages.wolfi.dev/os" >> /etc/apk/repositories

RUN for i in 1 2 3; do \
      apk add --no-cache bash openssl libssl3 tzdata nodejs python-3.13 libsndfile libatomic libevent nginx && break; \
      [ "$i" = 3 ] && { echo "apk add failed after 3 retries" >&2; exit 1; }; \
      sleep 5; \
    done
COPY --from=pgbouncer-builder /usr/local/bin/pgbouncer /usr/local/bin/pgbouncer

WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    LITELLM_NON_ROOT=true \
    PRISMA_BINARY_CACHE_DIR=/opt/prisma/binaries \
    PRISMA_CLI_PATH=/opt/prisma/binaries/node_modules/.bin/prisma \
    PRISMA_CLI_QUERY_ENGINE_TYPE=binary \
    PRISMA_SKIP_POSTINSTALL_GENERATE=1 \
    PRISMA_HIDE_UPDATE_MESSAGE=1 \
    PRISMA_ENGINES_CHECKSUM_IGNORE_MISSING=1 \
    PRISMA_OFFLINE_MODE=true

# Copy only what runtime needs. The application is installed inside the venv;
# the rest of the builder's /app is source and build metadata that must not
# ship (manifest-scanning tools attribute everything in it to this image).
COPY --from=builder /app/.venv /app/.venv
COPY --from=liteadmin-builder /opt/liteadmin /opt/liteadmin
COPY --from=builder /app/litellm/deploy/docker/docker-entrypoint.sh /app/docker-entrypoint.sh
COPY --from=builder /app/litellm/schema.prisma /app/litellm/schema.prisma
COPY --from=builder /app/litellm/schema.prisma /app/schema.prisma
COPY --from=builder /app/litellm/runtime/gateway /app/gateway
COPY --from=builder /app/litellm/proxy/prisma_migration.py /app/litellm/proxy/prisma_migration.py
# enterprise/ is imported by source path at runtime (proxy_cli puts the
# working directory on sys.path; litellm/proxy/hooks resolves
# litellm.enterprise.enterprise_hooks from it)
COPY --from=builder /opt/prisma /opt/prisma
COPY --from=builder /var/lib/litellm /var/lib/litellm
COPY litellm/ui/nginx.conf /etc/nginx/nginx.conf
COPY litellm/ui/docker-entrypoint.d/40-litellm-runtime-config.sh /app/ui/docker-entrypoint.d/40-litellm-runtime-config.sh

RUN find /app/.venv -depth -type d -path "*/tornado/test" -exec rm -rf {} + && \
    mkdir -p /app/.cache /var/run/litellm && \
    chown -R 65532:0 /app /var/lib/litellm /var/run/litellm && \
    chmod -R g=u,g+w /app/.cache /var/lib/litellm /var/run/litellm && \
    PRISMA_PATH="$(python -c 'import os, prisma; print(os.path.dirname(prisma.__file__))')" && \
    PROXY_EXTRAS_PATH="$(python -c 'import os, litellm_proxy_extras; print(os.path.dirname(litellm_proxy_extras.__file__))')" && \
    chmod -R g=u,g+w "$PRISMA_PATH" "$PROXY_EXTRAS_PATH" && \
    chmod -R a+rX /opt/prisma && \
    test -x /opt/prisma/binaries/node_modules/.bin/prisma && \
    test -f /opt/prisma/binaries/node_modules/prisma/build/index.js && \
    ls /opt/prisma/binaries/node_modules/@prisma/engines/query-engine-* >/dev/null && \
    python -c "from prisma.client import BINARY_PATHS; paths = list(BINARY_PATHS.query_engine.values()); assert paths and all(p.startswith('/opt/prisma/') for p in paths), paths" && \
    python -c "from litellm.rust_bridge import _native" && \
    python -c "import gateway.launch, litellm.runtime.backend.main"

USER 65532:65532

RUN nginx -t

EXPOSE 4000/tcp 4001/tcp 3000/tcp

ENTRYPOINT ["/app/docker-entrypoint.sh"]
