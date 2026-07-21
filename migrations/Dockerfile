ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:02dab76bd852a70556b5b2002195c8a5fdab77d323c433bf6642aab080489795
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:02dab76bd852a70556b5b2002195c8a5fdab77d323c433bf6642aab080489795
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM $UV_IMAGE AS uvbin

# ---------- Builder ----------
#
# Minimal install for `prisma migrate deploy`. We deliberately skip the heavy
# `proxy-runtime` (otel, sentry, ddtrace, pypdf, google-genai, anthropic-vertex,
# ...) and `semantic-router` extras that the gateway/backend pull in — the
# migration engine doesn't need them. We DO install `--extra proxy` so the
# DB-URL helper from `litellm.proxy.auth.rds_iam_token` is importable, which
# is how the gateway and backend assemble `DATABASE_URL` at pod startup when
# `IAM_TOKEN_DB_AUTH=true` (see backend/main.py:17, gateway/main.py:22). And
# `--extra extra_proxy` provides the `prisma` CLI + the secret-manager
# backends `litellm.secret_managers.main` lazily imports.
#
# `prisma generate` runs once at BUILD time to (a) install the Node-based
# Prisma CLI into the binary cache and (b) download the migration / query
# engine binaries. The Python client it also produces is unused by this
# image's runtime entrypoint — that's fine, it's a few hundred KB and the
# alternative (`prisma py fetch`) doesn't reliably trigger engine downloads
# under nodeenv. Crucially we do NOT run `prisma generate` at RUNTIME; the
# old migration job did, on every pod start, which is the wasteful behaviour
# the componentization is fixing.
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

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    PRISMA_USE_GLOBAL_NODE=true \
    PATH="/app/.venv/bin:${PATH}"

# Stage 1 — install third-party deps only (cached by pyproject.toml/uv.lock).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=enterprise/pyproject.toml,target=enterprise/pyproject.toml \
    --mount=type=bind,source=litellm-proxy-extras/pyproject.toml,target=litellm-proxy-extras/pyproject.toml \
    uv sync --frozen --no-install-project --no-install-workspace --no-default-groups --no-editable \
        --extra proxy \
        --extra extra_proxy \
        --python python3

# Stage 2 — copy source and install the project + workspace members.
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable \
        --extra proxy \
        --extra extra_proxy \
        --python python3

COPY migrations/run.py /app/run.py

# Pre-warm the Prisma binary cache so the Job pod doesn't reach the
# internet on first start. This matches what the backend Dockerfile does:
# `prisma generate` runs nodeenv (downloads Node), installs the prisma npm
# CLI, downloads the engine binaries for each `binaryTarget` in
# schema.prisma, AND emits the generated Python client. We don't need the
# client at runtime — the migration job invokes `prisma migrate deploy`
# via subprocess — but having it cached is harmless and the alternative
# (`prisma py fetch`) doesn't reliably trigger engine downloads.
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

# wolfi-base ships an unprivileged `nonroot` account (UID/GID 65532). The
# Prisma engine binaries are dynamically linked against libssl/libcrypto, so
# openssl stays in the runtime layer.
WORKDIR /app
ENV HOME=/home/nonroot \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder --chown=nonroot:nonroot /app /app
COPY --from=builder --chown=nonroot:nonroot /home/nonroot/.cache /home/nonroot/.cache

USER nonroot

ENTRYPOINT ["python3", "/app/run.py"]
