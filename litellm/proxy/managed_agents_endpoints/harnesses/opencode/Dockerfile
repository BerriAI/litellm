# ------------------------------------------------------------------ installer
FROM debian:bookworm-slim AS installer

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://opencode.ai/install | bash

# Drop install bundles / caches if any
RUN find /root/.opencode -maxdepth 2 -type d -name 'cache' -exec rm -rf {} + 2>/dev/null || true \
 && find /root/.opencode -maxdepth 2 -name '*.zip' -delete 2>/dev/null || true \
 && find /root/.opencode -maxdepth 2 -name '*.tar*' -delete 2>/dev/null || true

# ------------------------------------------------------------------ runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates bash python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/bash sandbox \
    && mkdir -p /work \
    && chown sandbox:sandbox /work

COPY --from=installer --chown=sandbox:sandbox /root/.opencode /home/sandbox/.opencode
ENV PATH="/home/sandbox/.opencode/bin:${PATH}"

WORKDIR /work
COPY --chown=sandbox:sandbox entrypoint.sh /entrypoint.sh
COPY --chown=sandbox:sandbox shim.py /shim.py
RUN chmod +x /entrypoint.sh /shim.py

USER sandbox

EXPOSE 4096 4097
ENTRYPOINT ["python3", "/shim.py"]
