#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf '用法：%s /绝对路径/litellm 后端基础镜像\n' "$0" >&2
  exit 2
fi

patch_root="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$1" && pwd)"
base_image="$2"
build_root="$(mktemp -d -t litellm-zh-cn.XXXXXXXX)"
trap 'rm -rf "$build_root"' EXIT

mkdir -p "$build_root/ui" "$build_root/runtime"
tar -C "$repo_root/ui/litellm-dashboard" --exclude=node_modules --exclude=.next --exclude=out -cf "$build_root/frontend.tar" .
tar -C "$build_root/ui" -xf "$build_root/frontend.tar"

docker run --rm \
  --mount "type=bind,source=$build_root,target=/work" \
  --mount "type=bind,source=$patch_root,target=/patch,readonly" \
  --env HTTPS_PROXY --env HTTP_PROXY --env NO_PROXY \
  --env NEXT_TELEMETRY_DISABLED=1 --env NODE_OPTIONS=--max-old-space-size=4096 \
  node:24.19-alpine3.24 sh -ec '
    cd /work/ui
    corepack enable
    corepack prepare pnpm@11.7.0 --activate
    if [ ! -f pnpm-workspace.yaml ]; then cp /patch/pnpm-workspace.yaml .; fi
    pnpm import
    pnpm install --frozen-lockfile --store-dir /work/pnpm-store
    pnpm rebuild esbuild unrs-resolver
    TYPESCRIPT_PATH=/work/ui/node_modules/typescript node --test /patch/tests/localize.test.mjs
    node /patch/localize.mjs apply --source /work/ui --catalog /patch/zh-CN.json --output /work/translation-report.json
    node /patch/prepare-ui.mjs /work/ui
    pnpm exec next build
  '

cp -R "$build_root/ui/out" "$build_root/runtime/ui"
cp "$patch_root/Dockerfile.runtime" "$build_root/runtime/Dockerfile"
docker build --build-arg "LITELLM_BASE_IMAGE=$base_image" --tag litellm-local:zh-cn "$build_root/runtime"
cp "$build_root/translation-report.json" "$patch_root/translation-report.json"
printf '中文镜像已构建为 litellm-local:zh-cn，翻译报告位于 %s/translation-report.json\n' "$patch_root"
