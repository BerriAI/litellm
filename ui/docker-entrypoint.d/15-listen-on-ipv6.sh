#!/bin/sh
# Renders the UI server's IPv6 listen directive, opt-in via NGINX_LISTEN_IPV6:
# "true" forces it on, "auto" enables it only when the container has an IPv6
# stack (same /proc/net/if_inet6 gate as the stock
# 10-listen-on-ipv6-by-default.sh), anything else keeps today's IPv4-only bind.

set -eu

ME=$(basename "$0")
SNIPPET="/etc/nginx/listen-ipv6/enabled.conf"

entrypoint_log() {
    if [ -z "${NGINX_ENTRYPOINT_QUIET_LOGS:-}" ]; then
        echo "$ME: $*"
    fi
}

case "${NGINX_LISTEN_IPV6:-}" in
    true|on|1)
        ;;
    auto)
        if [ ! -f /proc/net/if_inet6 ]; then
            entrypoint_log "info: NGINX_LISTEN_IPV6=auto and ipv6 not available, keeping IPv4 only"
            exit 0
        fi
        ;;
    *)
        exit 0
        ;;
esac

if ! touch "$SNIPPET" 2>/dev/null; then
    entrypoint_log "info: can not write $SNIPPET (read-only file system?), keeping IPv4 only"
    exit 0
fi

echo "listen [::]:3000 default_server;" > "$SNIPPET"
entrypoint_log "info: enabled listen on [::]:3000"
