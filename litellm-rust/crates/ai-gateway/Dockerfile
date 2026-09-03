# Multi-stage build for the LiteLLM Rust AI Gateway (realtime WebSocket proxy).
#
# Build context is the **repo root** so we can install `litellm` from this repo's
# source (the gateway loads its model_list via litellm.proxy.read_model_list,
# which is not in any PyPI release yet) AND build the rust workspace under
# litellm-rust/.
#
#   docker build -f litellm-rust/crates/ai-gateway/Dockerfile -t litellm-ai-gateway .
#
# No secrets live in this file. Runtime config (LITELLM_MASTER_KEY,
# OPENAI_API_KEY referenced by config.yaml, etc.) is injected as environment
# variables at deploy time.

# ---- Chef -------------------------------------------------------------------
# cargo-chef caches the dependency build so only the gateway crate recompiles on
# a source-only change. python3-dev is present in every rust stage because the
# `python-config` feature links libpython via pyo3 (even in the cook step), and
# python3-pip builds the litellm wheel in the builder stage.
FROM rust:1.98-slim-bookworm AS chef
ENV PYO3_PYTHON=python3.11
# rustup reads rust-toolchain.toml from any parent of the working directory, so
# copying it in is what keeps every cargo call below on the repo's pinned
# channel rather than on whatever the base image happens to ship.
COPY rust-toolchain.toml /build/rust-toolchain.toml
WORKDIR /build/litellm-rust
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip pkg-config libssl-dev clang \
    && rm -rf /var/lib/apt/lists/* \
    && cargo install cargo-chef --locked --version 0.1.77

# ---- Planner ----------------------------------------------------------------
# Produce the dependency recipe from the rust workspace manifests + Cargo.lock.
FROM chef AS planner
COPY litellm-rust/ .
RUN cargo chef prepare --recipe-path recipe.json

# ---- Builder ----------------------------------------------------------------
FROM chef AS builder
# Cook (compile) just the dependencies first — this layer is cached and reused
# whenever only gateway source changes.
COPY --from=planner /build/litellm-rust/recipe.json recipe.json
RUN cargo chef cook --locked --release \
        -p litellm-ai-gateway --features server,python-config \
        --recipe-path recipe.json
# Now copy the real sources and build the gateway binary. Deps are already cooked
# above, so this step only recompiles the gateway crate.
COPY litellm-rust/ .
RUN cargo build --locked --release -p litellm-ai-gateway --bin litellm-ai-gateway --features server,python-config

# The root pyproject builds with maturin against litellm-rust/crates/python-bridge,
# so the wheel is built here, next to the crate sources and the cargo toolchain,
# and the runtime stage installs the artifact instead of compiling anything.
COPY pyproject.toml README.md LICENSE /build/
COPY litellm/ /build/litellm/
COPY enterprise/ /build/enterprise/
RUN pip3 wheel --no-cache-dir --no-deps --wheel-dir /build/dist /build

# ---- Runtime ----------------------------------------------------------------
# python:3.11-slim-bookworm ships libpython3.11, matching the builder's PyO3
# 3.11 ABI so the embedded interpreter links and imports cleanly.
FROM python:3.11-slim-bookworm AS runtime

# CA certificates for outbound TLS to the OpenAI realtime endpoint.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install litellm (with proxy extras) FROM THIS REPO'S SOURCE so
# `import litellm.proxy.read_model_list` works — it is not on PyPI yet.
COPY --from=builder /build/dist/*.whl /tmp/wheels/
RUN wheel="$(ls /tmp/wheels/litellm-*.whl)" \
    && pip install --no-cache-dir "${wheel}[proxy]" \
    && rm -rf /tmp/wheels

# The compiled gateway binary (pure-Rust realtime hot path; Python is load-time
# only).
COPY --from=builder /build/litellm-rust/target/release/litellm-ai-gateway /usr/local/bin/litellm-ai-gateway

# Default config.yaml. A real deploy can override this (e.g. mount a Render
# secret file at the same path) — never bake secrets into the image.
COPY litellm-rust/crates/ai-gateway/config.yaml /app/config.yaml

# Bind to all interfaces (Render routes to 0.0.0.0:$PORT) and load the model_list
# from config.yaml via the embedded python config reader.
ENV HOST=0.0.0.0 \
    LITELLM_CONFIG_PATH=/app/config.yaml

# Drop to a non-root user. The realtime hot path needs no root privileges, so
# running unprivileged limits blast radius if the process is ever compromised.
# The binary in /usr/local/bin is world-executable (COPY default mode 755); we
# only need /app (and the config.yaml it reads) owned by the unprivileged user.
RUN useradd --system --no-create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/usr/local/bin/litellm-ai-gateway"]
