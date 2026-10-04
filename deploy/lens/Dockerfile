ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:1d95114038f76513a9ace6fca107d5582b08c65981f81f61cb56bf7fd2ef216d
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:1d95114038f76513a9ace6fca107d5582b08c65981f81f61cb56bf7fd2ef216d
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM $UV_IMAGE AS uvbin

FROM $LITELLM_BUILD_IMAGE AS builder
COPY --from=uvbin /uv /usr/local/bin/uv
RUN apk add --no-cache python-3.13
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy
WORKDIR /app
COPY deploy/lens/requirements.lock /tmp/requirements.lock
RUN uv venv --python python3.13 /app/.venv && \
    uv pip sync --python /app/.venv/bin/python --require-hashes --only-binary :all: /tmp/requirements.lock

FROM $LITELLM_RUNTIME_IMAGE AS runtime
ARG LITELLM_RELEASE_TAG=""
RUN : "${LITELLM_RELEASE_TAG:?Pass --build-arg LITELLM_RELEASE_TAG matching the gateway}"
RUN apk add --no-cache python-3.13
ENV LITELLM_RELEASE_TAG=${LITELLM_RELEASE_TAG} \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY litellm/proxy/lens/__init__.py litellm/proxy/lens/models.py litellm/proxy/lens/trace_store.py litellm/proxy/lens/analysis.py litellm/proxy/lens/worker.py litellm/proxy/lens/release.py /app/lens/
COPY litellm/proxy/lens/prompts/ /app/lens/prompts/
USER 65532:65532
CMD ["python", "-m", "lens.worker"]
