FROM node:22-bookworm-slim AS frontend

WORKDIR /build/playground
COPY playground/package.json playground/package-lock.json ./
RUN npm ci
COPY playground/tsconfig.json ./tsconfig.json
COPY playground/index.html ./index.html
COPY playground/src ./src
RUN npm run build

FROM rust:1.98.0-bookworm AS rust-toolchain
RUN rustup component add rust-analyzer

FROM node:22-bookworm-slim

ARG OPENCODE_VERSION=1.18.30

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates clang cmake git libssl-dev pkg-config tini util-linux-extra \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd workspace \
    && useradd --create-home --gid workspace opencode \
    && useradd --create-home --gid workspace playground \
    && npm install --global "opencode-ai@${OPENCODE_VERSION}"

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV CARGO_HOME="/usr/local/cargo"

WORKDIR /opt/playground
COPY --from=frontend /build/playground/node_modules ./node_modules
COPY --from=frontend /build/playground/dist ./dist
COPY playground/package.json playground/package-lock.json playground/server.mjs ./
COPY playground/deploy/opencode.json /etc/opencode/opencode.json
COPY playground/deploy/xdg-open /usr/local/bin/xdg-open
COPY playground/deploy/start.sh /usr/local/bin/start-playground
COPY litellm-rust /opt/litellm-seed/litellm-rust
COPY playground/workspace /opt/litellm-seed/playground/workspace

RUN chmod 0755 /usr/local/bin/start-playground /usr/local/bin/xdg-open

EXPOSE 8080 4096

ENTRYPOINT ["tini", "--", "/usr/local/bin/start-playground"]
