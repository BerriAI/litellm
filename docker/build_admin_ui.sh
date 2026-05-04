#!/bin/bash

# Builds the Admin UI Next.js bundle and publishes it to
# litellm/proxy/_experimental/out so the proxy can serve it.
#
# This script is invoked from every UI-bearing Dockerfile and from the
# release pipeline. The pre-built bundle is no longer checked into the
# repository, so this script must run successfully whenever the
# resulting Docker image / Python wheel needs to ship the dashboard.

set -e

pwd

# Apply the enterprise color palette when present so OSS and enterprise
# images can share this build script.
if [ -f "enterprise/enterprise_ui/enterprise_colors.json" ]; then
    echo "Using enterprise UI color palette"
    cp enterprise/enterprise_ui/enterprise_colors.json ui/litellm-dashboard/ui_colors.json
else
    echo "Using default LiteLLM UI color palette"
fi

cd ui/litellm-dashboard

# Use a deterministic install when a lockfile is present.
if [ -f package-lock.json ]; then
    npm ci
else
    npm install
fi

npm run build

destination_dir="../../litellm/proxy/_experimental/out"
mkdir -p "$destination_dir"
rm -rf "$destination_dir"/*
cp -r ./out/. "$destination_dir"/
rm -rf ./out

# Restructure HTML so extensionless routes work (e.g. /ui/login).
# Next.js export produces login.html; the proxy expects login/index.html.
find "$destination_dir" -name '*.html' ! -name 'index.html' | while read -r htmlfile; do
    target_dir="${htmlfile%.html}"
    mkdir -p "$target_dir"
    mv "$htmlfile" "$target_dir/index.html"
done

cd ../..
