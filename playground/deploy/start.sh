#!/usr/bin/env bash
set -euo pipefail

: "${OPENCODE_SERVER_PASSWORD:?Set OPENCODE_SERVER_PASSWORD as a Fly secret}"
umask 0002

volume_root="${PLAYGROUND_VOLUME_ROOT:-/data}"
repository_directory="${volume_root}/workspace/litellm"
workspace_directory="${repository_directory}/playground/workspace"
opencode_data_directory="${volume_root}/opencode"
cargo_cache_directory="${volume_root}/cache/cargo"

mkdir -p "${repository_directory}" "${opencode_data_directory}" "${cargo_cache_directory}" /home/playground
cp -a -n /opt/litellm-seed/. "${repository_directory}/"

if [[ ! -d "${repository_directory}/.git" ]]; then
  git -C "${repository_directory}" init --quiet --initial-branch=main
  git -C "${repository_directory}" config user.name "Playground"
  git -C "${repository_directory}" config user.email "playground@localhost"
  git -C "${repository_directory}" add .
  git -C "${repository_directory}" commit --quiet -m "Initial volume workspace"
fi

chown -R opencode:workspace "${opencode_data_directory}"
chown -R playground:workspace "${cargo_cache_directory}"
chown -R root:workspace "${repository_directory}"
chmod -R g+rwX "${repository_directory}"
find "${repository_directory}" -type d -exec chmod g+s {} +
git config --system --get-all safe.directory | grep -Fxq "${repository_directory}" \
  || git config --system --add safe.directory "${repository_directory}"

(
  cd "${repository_directory}"
  exec runuser -u opencode --preserve-environment -- env \
    HOME=/home/opencode \
    BROWSER=/bin/true \
    XDG_DATA_HOME="${opencode_data_directory}" \
    OPENCODE_CONFIG=/etc/opencode/opencode.json \
    opencode web --hostname 0.0.0.0 --port 4096
) &
opencode_pid=$!

runuser -u playground -- env -i \
  PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME=/home/playground \
  CARGO_HOME="${cargo_cache_directory}" \
  RUSTUP_HOME=/usr/local/rustup \
  NODE_ENV=production \
  HOST=0.0.0.0 \
  PORT=8080 \
  PLAYGROUND_PASSWORD="${OPENCODE_SERVER_PASSWORD}" \
  PLAYGROUND_REPOSITORY_DIR="${repository_directory}" \
  PLAYGROUND_WORKSPACE_DIR="${workspace_directory}" \
  node /opt/playground/server.mjs &
playground_pid=$!

trap 'kill "${opencode_pid}" "${playground_pid}" 2>/dev/null || true' EXIT INT TERM
wait -n "${opencode_pid}" "${playground_pid}"
