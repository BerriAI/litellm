#!/usr/bin/env bash
# Build the Rust gateway image and the load-test tools image from the current checkout, tagged
# with the commit so every perf result maps back to a build.
#
#   scripts/build-images.sh                  # fat LTO, what ships (slow: ~15 min cold)
#   FAST=1 scripts/build-images.sh           # no LTO, for iterating (several times faster)
#   PUSH=1 REGISTRY=123.dkr.ecr.us-east-1.amazonaws.com/litellm scripts/build-images.sh
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
rust="$(cd "$here/.." && pwd)"
sha="$(git -C "$rust" rev-parse --short HEAD)"
dirty="$(git -C "$rust" diff --quiet HEAD -- . || echo -dirty)"
tag="${TAG:-customer-proxy-$sha$dirty}"
registry="${REGISTRY:-}"
prefix="${registry:+$registry/}"

lto_args=()
if [[ "${FAST:-0}" == 1 ]]; then
  lto_args=(--build-arg LTO=off --build-arg CODEGEN_UNITS=16)
  tag="$tag-nolto"
fi

gateway="${prefix}litellm-rust-gateway:$tag"
tools="${prefix}litellm-loadtest:$tag"

docker build "${lto_args[@]}" -f "$rust/crates/gateway/Dockerfile" -t "$gateway" "$rust"
docker build -t "$tools" "$here"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$gateway"
  docker push "$tools"
fi
echo "gateway image: $gateway"
echo "tools image:   $tools"
