FROM rust:1.98.0-bookworm@sha256:82150a52ec202c1b14d7817e14516c392bb7f5cfebd88f1ed531cb37ebd39922 AS chef

RUN apt-get update \
    && apt-get install --yes --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install cargo-chef --version 0.1.74 --locked

WORKDIR /app

FROM chef AS planner

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

COPY --from=planner /app/recipe.json ./recipe.json

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    cargo chef cook --release --locked --package litellm-gateway --bin litellm-gateway --recipe-path recipe.json

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    cargo build --release --locked --package litellm-gateway --bin litellm-gateway

FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251 AS runtime

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 litellm \
    && useradd --uid 10001 --gid litellm --no-create-home --shell /usr/sbin/nologin litellm

WORKDIR /app

COPY --from=builder /app/target/release/litellm-gateway /usr/local/bin/litellm-gateway

USER 10001:10001

EXPOSE 4000

ENTRYPOINT ["/usr/local/bin/litellm-gateway"]
