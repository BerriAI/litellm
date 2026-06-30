FROM python:3.12-slim-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Download rustup-init and verify its checksum using the official sidecar.
# Pinned to rustup 1.28.1 on x86_64 linux.
RUN curl -Lo /tmp/rustup-init \
      https://static.rust-lang.org/rustup/archive/1.28.1/x86_64-unknown-linux-gnu/rustup-init \
    && curl -Lo /tmp/rustup-init.sha256 \
      https://static.rust-lang.org/rustup/archive/1.28.1/x86_64-unknown-linux-gnu/rustup-init.sha256 \
    && sha256sum --check /tmp/rustup-init.sha256 \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --profile minimal --default-toolchain 1.87.0 --no-modify-path \
    && rm /tmp/rustup-init /tmp/rustup-init.sha256

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir "headroom-ai[proxy]==0.27.0"


FROM python:3.12-slim-bookworm

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/headroom /usr/local/bin/headroom

EXPOSE 8787

ENV HEADROOM_TELEMETRY=off

CMD ["headroom", "proxy", "--host", "0.0.0.0", "--port", "8787", "--no-optimize"]
