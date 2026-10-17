#!/bin/sh
set -eu

if [ "${1:-}" = nginx ] && [ -n "${SERVER_ROOT_PATH:-}" ] && [ "$SERVER_ROOT_PATH" != / ]; then
    runtime_dir=$(mktemp -d /tmp/litellm-ui.XXXXXX)
    /usr/local/bin/prepare-ui-root-path /usr/share/nginx/html "$runtime_dir" \
        /etc/nginx/nginx.conf "$runtime_dir/nginx.conf" "$SERVER_ROOT_PATH"
    shift
    set -- nginx -c "$runtime_dir/nginx.conf" "$@"
fi

exec /docker-entrypoint.sh "$@"
