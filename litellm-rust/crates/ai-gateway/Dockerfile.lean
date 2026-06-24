# Lean multi-stage build for the LiteLLM Rust AI Gateway (realtime WebSocket proxy).
#
# This is the FAST, MINIMAL image used for load testing the realtime connection
# pool. Unlike the python-config Dockerfile, it builds the gateway with DEFAULT
# features (no `python-config`), so it does NOT link libpython or pip-install
# litellm. The model_list is built from the environment stand-in path in
# `build_router_from_env()` (OPENAI_REALTIME_MODEL + OPENAI_API_KEY).
#
#   docker build -f litellm-rust/crates/ai-gateway/Dockerfile.lean \
#     -t litellm-ai-gateway-lean .
#
# Build context is the repo root (Render convention) but only litellm-rust/ is
# actually needed — see Dockerfile.lean.dockerignore.
#
# No secrets live in this file. Runtime config (LITELLM_MASTER_KEY,
# OPENAI_API_KEY, REALTIME_POOL_SIZE, OPENAI_REALTIME_MODEL) is injected as
# environment variables at deploy time.

# ---- Chef -------------------------------------------------------------------
# cargo-chef caches the dependency build so only the gateway crate recompiles on
# a source-only change.
FROM rust:1.90-slim-bookworm AS chef
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cargo install cargo-chef --locked --version 0.1.77
WORKDIR /build/litellm-rust

# ---- Planner ----------------------------------------------------------------
FROM chef AS planner
COPY litellm-rust/ .
RUN cargo chef prepare --recipe-path recipe.json

# ---- Builder ----------------------------------------------------------------
FROM chef AS builder
# Cook (compile) just the dependencies first — cached unless deps change. No
# `--features python-config`, so libpython is never linked.
COPY --from=planner /build/litellm-rust/recipe.json recipe.json
RUN cargo chef cook --locked --release \
        -p litellm-ai-gateway \
        --recipe-path recipe.json
COPY litellm-rust/ .
RUN cargo build --locked --release -p litellm-ai-gateway

# ---- Runtime ----------------------------------------------------------------
# A tiny Debian slim base (no python). Only needs a CA bundle for outbound TLS to
# the OpenAI realtime endpoint.
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/litellm-rust/target/release/litellm-ai-gateway /usr/local/bin/litellm-ai-gateway

# Bind to all interfaces (Render routes to 0.0.0.0:$PORT). The env stand-in
# router uses OPENAI_REALTIME_MODEL + OPENAI_API_KEY; the pool reads
# REALTIME_POOL_SIZE. All injected at deploy time.
ENV HOST=0.0.0.0

# Drop to a non-root user.
RUN useradd --system --no-create-home --uid 10001 appuser
USER appuser

ENTRYPOINT ["/usr/local/bin/litellm-ai-gateway"]
