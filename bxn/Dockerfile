FROM ghcr.io/bxnlabs/containers/py-devtools:20250217.3_0ea39c2@sha256:004395e7ccd2dd424bf525ba313201b3bfe76949c089985700bb22f3e8543b6b AS devtools

# Dependencies
COPY pyproject.toml [p]oetry.lock ./
RUN --mount=type=cache,target=/var/cache/pypoetry,sharing=locked \
    POETRY_CACHE_DIR=/var/cache/pypoetry uvx poetry install --sync --no-root --extras=proxy --compile

# Source
COPY README.md ./
COPY enterprise ./enterprise
COPY litellm ./litellm
RUN uvx poetry install --only-root --compile


FROM devtools AS build

# Prune dependencies to only what we need for release
RUN --mount=type=cache,target=/var/cache/pypoetry,sharing=locked \
    POETRY_CACHE_DIR=/var/cache/pypoetry uvx poetry install --only=main --sync --no-root --extras=proxy --compile
RUN uv pip install --no-deps --compile .


FROM ghcr.io/bxnlabs/containers/py-base:20250217.3_0ea39c2@sha256:868480f70b04f8782bcc2eb5b1f2b074a4ebaf2b758b18936cc5d2f79dc12117

COPY --from=build ${VIRTUAL_ENV}/ ${VIRTUAL_ENV}/
