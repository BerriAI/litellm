#!/usr/bin/env sh
# Fetch the recording proxy's CA cert and add it to the local trust store
# *and* every Python TLS client we know about. Idempotent.
#
# Why so many trust-store paths? Different clients in litellm read different
# CA bundles:
#
#   - openssl / curl / requests / aiohttp default: /etc/ssl/certs/ca-certificates.crt
#     (Debian/Wolfi) or the OpenSSL default (Alpine)
#   - boto3 / botocore: $AWS_CA_BUNDLE if set, else certifi
#   - httpx: certifi by default unless SSL_CERT_FILE points elsewhere
#   - openai-python: certifi (httpx)
#   - google.auth: certifi (httpx) or system roots depending on transport
#
# We update certifi in-place inside the litellm venv and additionally export
# SSL_CERT_FILE / REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE so anything that
# honors the environment finds it.
#
# Caller is expected to:
#   1) export PROXY_HOST / PROXY_PORT (default cassette-proxy:8080)
#   2) source this script (so the env vars persist), or eval its output

set -eu

PROXY_HOST="${PROXY_HOST:-cassette-proxy}"
PROXY_PORT="${PROXY_PORT:-8080}"
CA_PATH="${CA_PATH:-/usr/local/share/ca-certificates/litellm-cassette-proxy.crt}"
CA_FETCH_ENDPOINT="http://${PROXY_HOST}:${PROXY_PORT}/cert/pem"

mkdir -p "$(dirname "$CA_PATH")"

# mitmproxy serves its CA at /cert/pem when it sees a plain HTTP request to
# any host (the magic bypasses the CONNECT path).
if ! curl --silent --show-error --max-time 30 \
        --proxy "http://${PROXY_HOST}:${PROXY_PORT}" \
        -o "$CA_PATH" "$CA_FETCH_ENDPOINT"; then
    echo "trust_ca.sh: could not fetch CA from $CA_FETCH_ENDPOINT" >&2
    exit 1
fi

# Update system trust store. Best-effort across distros.
if command -v update-ca-certificates >/dev/null 2>&1; then
    update-ca-certificates --fresh >/dev/null 2>&1 || true
elif command -v update-ca-trust >/dev/null 2>&1; then
    cp "$CA_PATH" /etc/pki/ca-trust/source/anchors/ 2>/dev/null || true
    update-ca-trust 2>/dev/null || true
fi

# Update certifi in *every* Python venv we can find (litellm runs out of
# /app/.venv in the database image, but be permissive).
for cacert in $(find / -name cacert.pem 2>/dev/null); do
    # Only append if our CA isn't already in the bundle.
    if ! grep -q -F "$(head -n 2 "$CA_PATH")" "$cacert" 2>/dev/null; then
        cat "$CA_PATH" >> "$cacert"
    fi
done

# Export for downstream processes. The caller should source this script
# (or otherwise propagate these env vars) for them to take effect.
export HTTP_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
export HTTPS_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,host.docker.internal}"
export SSL_CERT_FILE="$CA_PATH"
export REQUESTS_CA_BUNDLE="$CA_PATH"
export CURL_CA_BUNDLE="$CA_PATH"
export AWS_CA_BUNDLE="$CA_PATH"
export NODE_EXTRA_CA_CERTS="$CA_PATH"

echo "trust_ca.sh: trusted CA at $CA_PATH; routing egress via $HTTPS_PROXY"
