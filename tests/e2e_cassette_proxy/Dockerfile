# Recording HTTP proxy used by e2e CI jobs.
#
# Pinned to ``python:3.12-slim`` rather than the official mitmproxy image
# because we need:
#   - msgpack (binary wheels not always present in the upstream image)
#   - redis-py
#   - the addon source from this repo
# and pinning + verifying the upstream mitmproxy image would force us to
# carry yet another sha256 in CI. Building from a known python base
# keeps the supply chain to one tag we already control.
#
# Image is intentionally fat; it only runs in CI sidecars.

FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Pin every dependency.
RUN pip install \
        "mitmproxy==11.0.2" \
        "redis==5.2.0" \
        "msgpack==1.1.0"

WORKDIR /app

COPY tests/e2e_cassette_proxy /app/tests/e2e_cassette_proxy

ENV PYTHONPATH=/app

VOLUME ["/root/.mitmproxy"]

EXPOSE 8080

# Args explained in tests/e2e_cassette_proxy/README.md.
#
# We run as root inside the container so VOLUME ["/root/.mitmproxy"] works
# without uid juggling — this image only runs in CI sidecars, not in
# production.
ENTRYPOINT ["mitmdump", \
    "--listen-host", "0.0.0.0", \
    "--listen-port", "8080", \
    "--set", "block_global=false", \
    "--set", "ssl_insecure=true", \
    "--set", "termlog_verbosity=info", \
    "-s", "/app/tests/e2e_cassette_proxy/addon.py"]
