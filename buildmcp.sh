#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

MCP_ROOT="mcp-servers"
BUILD_ROOT=".mcp-build"

MCP_DIRS=(
    "$MCP_ROOT/google-drive-mcp"
)

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

git submodule update --init --recursive

for mcp_dir in "${MCP_DIRS[@]}"; do
    mcp_name="$(basename "$mcp_dir")"
    artifact_dir="$BUILD_ROOT/$mcp_name"

    echo "Building $mcp_name"

    npm --prefix "$mcp_dir" install \
        --include=dev \
        --ignore-scripts \
        --no-audit \
        --no-fund 

    npm --prefix "$mcp_dir" run build
    npm --prefix "$mcp_dir" prune --omit=dev --ignore-scripts

    mkdir -p "$artifact_dir"

    cp "$mcp_dir/package.json" "$artifact_dir/package.json"
    cp "$mcp_dir/package-lock.json" "$artifact_dir/package-lock.json"
    cp -a "$mcp_dir/node_modules" "$artifact_dir/node_modules"
    cp -a "$mcp_dir/dist" "$artifact_dir/dist"
done
