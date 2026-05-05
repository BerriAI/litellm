# Base image for building
ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:f26d42a15d09d9a643b231df929fa3cf609bedc58a728eb445be89a9d8d1da9f

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:f26d42a15d09d9a643b231df929fa3cf609bedc58a728eb445be89a9d8d1da9f
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:733b4042187702f832f7fdecb3aff14a61b288c4ca37af188bb5715c1caebaf8

FROM $UV_IMAGE AS uvbin

# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

WORKDIR /app
USER root

COPY --from=uvbin /uv /usr/local/bin/uv
COPY --from=uvbin /uvx /usr/local/bin/uvx

RUN apk add --no-cache \
    bash \
    gcc \
    python3 \
    python3-dev \
    openssl \
    openssl-dev \
    nodejs \
    npm \
    libsndfile

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

# Copy dependency metadata first for layer caching
COPY pyproject.toml uv.lock ./
COPY enterprise/pyproject.toml enterprise/
COPY litellm-proxy-extras/pyproject.toml litellm-proxy-extras/

# Install third-party dependencies (cached unless pyproject.toml/uv.lock change)
RUN uv sync --frozen --no-install-project --no-install-workspace --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python python3

# Copy full source tree
COPY . .

# Build Admin UI before final sync
RUN sed -i 's/\r$//' docker/build_admin_ui.sh && chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Install project and workspace packages (fast - deps already cached)
RUN uv sync --frozen --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python python3

RUN prisma generate --schema=./schema.prisma

RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh && \
    sed -i 's/\r$//' docker/prod_entrypoint.sh && chmod +x docker/prod_entrypoint.sh

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime

USER root

RUN apk add --no-cache bash openssl tzdata nodejs npm python3 libsndfile supervisor && \
    npm install -g npm@11.12.1 tar@7.5.11 glob@13.0.6 @isaacs/brace-expansion@5.0.1 brace-expansion@5.0.5 minimatch@10.2.4 diff@8.0.3 picomatch@4.0.4 && \
    GLOBAL="$(npm root -g)" && \
    for pkg in tar glob @isaacs/brace-expansion brace-expansion minimatch diff picomatch; do \
        name="${pkg##*/}"; \
        find "$GLOBAL/npm" -type d -name "$name" -path "*/node_modules/$pkg" | while read d; do \
            rm -rf "$d" && cp -rL "$GLOBAL/$pkg" "$d"; \
        done; \
    done && \
    npm cache clean --force && \
    { apk del --no-cache npm 2>/dev/null || true; }

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"

COPY --from=builder /app /app
# Prisma binaries live in $HOME/.cache (default prisma-python location),
# which is /root/.cache here. Copy them from the builder so they survive
# deployments that volume-mount /app/.cache (e.g. readOnlyRootFilesystem
# + emptyDir) — otherwise the mount would shadow the baked-in query engine.
COPY --from=builder /root/.cache /root/.cache

RUN find /app/.venv -type f -path "*/tornado/test/*" -delete && \
    find /app/.venv -type d -path "*/tornado/test" -delete

EXPOSE 4000/tcp

COPY docker/supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["docker/prod_entrypoint.sh"]
CMD ["--port", "4000"]
