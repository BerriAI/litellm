ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:81211e15a86ea1d5357f284482af45730c51de45eaf10ad9ce205d6bfc2df323
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:81211e15a86ea1d5357f284482af45730c51de45eaf10ad9ce205d6bfc2df323
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM $UV_IMAGE AS uvbin

# ---------- Rust extension ----------
# Copies only the native build's inputs, so the fat-LTO compile stays cached across Python-only changes
FROM $LITELLM_BUILD_IMAGE AS rust-native

WORKDIR /app
USER root

COPY --from=uvbin /uv /uvx /usr/local/bin/

RUN for i in 1 2 3; do \
      apk add --no-cache bash gcc python-3.13 python-3.13-dev openssl openssl-dev && break; \
      [ $i = 3 ] && { echo "apk add failed after 3 retries" >&2; exit 1; }; \
      sleep 5; \
    done

ENV UV_PYTHON_DOWNLOADS=0

COPY pyproject.toml README.md rust-toolchain.toml ./
COPY .cargo ./.cargo
COPY litellm-rust ./litellm-rust

RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p litellm/rust_bridge /wheels && \
    touch litellm/__init__.py litellm/rust_bridge/__init__.py && \
    uvx --python python3.13 --from maturin==1.15.0 --with puccinialin==0.1.14 \
        python -c "import maturin; maturin.build_wheel('/wheels')"

# ---------- Builder ----------
FROM $LITELLM_BUILD_IMAGE AS builder

WORKDIR /app
USER root

COPY --from=uvbin /uv /uvx /usr/local/bin/

# nodejs/npm so `prisma generate` uses Wolfi's Node via PRISMA_USE_GLOBAL_NODE
# instead of nodeenv downloading one whose dynamic deps may not be in Wolfi
# (e.g. Node 26.2.0 needs libatomic). Retry for transient apk.cgr.dev flakes.
RUN for i in 1 2 3; do \
      apk add --no-cache bash gcc python-3.13 python-3.13-dev openssl libssl3 openssl-dev libsndfile nodejs npm && break; \
      [ $i = 3 ] && { echo "apk add failed after 3 retries" >&2; exit 1; }; \
      sleep 5; \
    done

# UV_COMPILE_BYTECODE=1   precompiles .pyc at install time → faster cold start.
# UV_LINK_MODE=copy       avoids hardlink warnings when uv installs from a
#                         BuildKit cache mount (different filesystem).
# UV_PYTHON_DOWNLOADS=0   force uv to use the apk-installed CPython instead of
#                         silently pulling a managed interpreter.
# PRISMA_USE_GLOBAL_NODE  explicit (matches default) so an env override can't
#                         silently re-enable nodeenv's Node download.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    PRISMA_USE_GLOBAL_NODE=true \
    PATH="/app/.venv/bin:${PATH}"

# Stage 1 — install dependencies only.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=enterprise/pyproject.toml,target=enterprise/pyproject.toml \
    --mount=type=bind,source=litellm-proxy-extras/pyproject.toml,target=litellm-proxy-extras/pyproject.toml \
    uv sync --frozen --no-install-project --no-install-workspace --no-default-groups --no-editable \
        --extra proxy \
        --extra proxy-runtime \
        --group admin-mcp \
        --extra extra_proxy \
        --extra semantic-router \
        --extra saml \
        --python python3.13

# Stage 2 — copy source and install the project + workspace members.
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-default-groups --no-editable \
        --extra proxy \
        --extra proxy-runtime \
        --group admin-mcp \
        --extra extra_proxy \
        --extra semantic-router \
        --extra saml \
        --python python3.13

RUN --mount=type=bind,from=rust-native,source=/wheels,target=/wheels \
    uv pip install --python /app/.venv/bin/python --no-deps /wheels/litellm-*.whl

RUN cp "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"/litellm/rust_bridge/_native*.so litellm/rust_bridge/

RUN HOME=/opt/prisma XDG_CACHE_HOME=/opt/prisma/.cache PRISMA_BINARY_CACHE_DIR=/opt/prisma/binaries \
    npm_config_cache=/root/.npm \
    prisma generate --schema=./schema.prisma

RUN sed -i 's/\r$//' docker/component_entrypoint.sh && chmod +x docker/component_entrypoint.sh

# ---------- Runtime ----------
FROM $LITELLM_RUNTIME_IMAGE AS runtime
ARG LITELLM_RELEASE_TAG=""
ENV LITELLM_RELEASE_TAG=${LITELLM_RELEASE_TAG}

USER root

RUN for i in 1 2 3; do \
      apk add --no-cache bash openssl libssl3 tzdata python-3.13 libsndfile libatomic && break; \
      [ $i = 3 ] && { echo "apk add failed after 3 retries" >&2; exit 1; }; \
      sleep 5; \
    done

# wolfi-base ships an unprivileged `nonroot` account (UID/GID 65532) with
# /home/nonroot. We run the backend as that user
WORKDIR /app
ENV HOME=/home/nonroot \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PRISMA_BINARY_CACHE_DIR=/opt/prisma/binaries

COPY --from=builder --chown=nonroot:nonroot /app /app
COPY --from=builder /opt/prisma /opt/prisma

RUN find /app/.venv -type f -path "*/tornado/test/*" -delete && \
    find /app/.venv -type d -path "*/tornado/test" -delete && \
    chmod -R a+rX /opt/prisma && \
    python -c "from prisma.client import BINARY_PATHS; paths = list(BINARY_PATHS.query_engine.values()); assert paths and all(p.startswith('/opt/prisma/') for p in paths), paths"

USER nonroot

EXPOSE 4001/tcp

ENTRYPOINT ["/app/docker/component_entrypoint.sh", "uvicorn", "backend.main:app"]
CMD ["--host", "0.0.0.0", "--port", "4001"]
