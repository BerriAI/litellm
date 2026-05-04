#!/bin/bash
set -e

# This script used to rebuild the UI and commit the resulting Next.js
# bundle into git so the proxy could ship a pre-built dashboard.
#
# The bundle is no longer checked into the repository — it is rebuilt
# from source by the Dockerfiles and by the PyPI publish workflow. This
# script now simply triggers a local build (the canonical entry point is
# `docker/build_admin_ui.sh`) for developers who want to preview the
# production-flavoured UI without booting a Docker image.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec "$repo_root/docker/build_admin_ui.sh"
