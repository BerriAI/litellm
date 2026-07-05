ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:c61ac6919b811ea53c4782d69f1fe05218ba3c25d53f01b6ab7892e621bd4370
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:c61ac6919b811ea53c4782d69f1fe05218ba3c25d53f01b6ab7892e621bd4370
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM $UV_IMAGE AS uvbin

# ---------- Builder ----------
FROM $LITELLM_BUILD_IMAGE AS builder

WORKDIR /app
USER root

COPY --from=uvbin /uv /uvx /usr/local/bin/

# nodejs/npm so `prisma generate` uses Wolfi's Node via PRISMA_USE_GLOBAL_NODE
# instead of nodeenv downloading one whose dynamic deps may not be in Wolfi
# (e.g. Node 26.2.0 needs libatomic). Retry for transient apk.cgr.dev flakes.
RUN for i in 1 2 3; do \
      apk add --no-cache bash gcc python3 python3-dev openssl openssl-dev libsndfile nodejs npm && break; \
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
        --extra extra_proxy \
        --extra semantic-router \
        --python python3

# Stage 2 — copy source and install the project + workspace members.
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable \
        --extra proxy \
        --extra proxy-runtime \
        --extra extra_proxy \
        --extra semantic-router \
        --python python3

RUN mkdir -p /home/nonroot && \
    HOME=/home/nonroot prisma generate --schema=./schema.prisma && \
    chown -R nonroot:nonroot /home/nonroot/.cache

# ---------- Runtime ----------
FROM $LITELLM_RUNTIME_IMAGE AS runtime

USER root

RUN for i in 1 2 3; do \
      apk add --no-cache bash openssl tzdata python3 libsndfile libatomic && break; \
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
    PYTHONUNBUFFERED=1

COPY --from=builder --chown=nonroot:nonroot /app /app
COPY --from=builder --chown=nonroot:nonroot /home/nonroot/.cache /home/nonroot/.cache

RUN find /app/.venv -type f -path "*/tornado/test/*" -delete && \
    find /app/.venv -type d -path "*/tornado/test" -delete

USER nonroot

EXPOSE 4001/tcp

ENTRYPOINT ["uvicorn", "backend.main:app"]
CMD ["--host", "0.0.0.0", "--port", "4001"]
