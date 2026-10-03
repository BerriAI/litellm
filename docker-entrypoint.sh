#!/bin/sh
# LiteLLM image entrypoint: docker-entrypoint.sh [COMPONENT] [ARGS...]
#
# COMPONENT selects the process this container runs; it can also be given as
# LITELLM_COMPONENT when the command carries no component. Anything after it is
# passed to that process. A first argument starting with "-" (or no argument at
# all) keeps the historical behaviour of running the monolithic proxy.
#
#   proxy       litellm ARGS                     everything in one process (default)
#   gateway     python -m gateway.launch ARGS    inference routes, 0.0.0.0:4000
#   backend     uvicorn backend.main:app ARGS    management routes, 0.0.0.0:4001
#   ui          nginx serving the admin UI       port 3000
#   migrations  python migrations/run.py         prisma migrate deploy, then exit
#   metrics     python -m litellm.proxy.prometheus_metrics_server ARGS
#   collector   python -m litellm.proxy.collector ARGS
#
# gateway and backend get their host and port defaults first, so ARGS such as
# --port 8080 override them. An unknown first word is executed as-is (docker run
# <image> sh). PgBouncer is not a component: LITELLM_PGBOUNCER_ENABLED=true
# starts it inside proxy and gateway. Components that write Prometheus samples
# start with an empty PROMETHEUS_MULTIPROC_DIR; metrics, collector and raw
# readers keep the workers' files, other raw commands clear stale samples.
set -eu

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//' >&2
}

component="${LITELLM_COMPONENT:-proxy}"
case "${1:-}" in
    proxy|gateway|backend|ui|migrations|metrics|collector)
        component="$1"
        shift
        ;;
    ""|-*)
        ;;
    help)
        usage
        exit 0
        ;;
    *)
        if [ -n "${PROMETHEUS_MULTIPROC_DIR:-}" ]; then
            mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
            case "$*" in
                *litellm.proxy.prometheus_metrics_server*|*litellm.proxy.collector*) ;;
                *) rm -f "$PROMETHEUS_MULTIPROC_DIR"/*.db ;;
            esac
        fi
        exec "$@"
        ;;
esac

case "$component" in
    proxy)
        set -- litellm "$@"
        ;;
    gateway)
        set -- python -m gateway.launch --workers "${NUM_WORKERS:-1}" --host 0.0.0.0 --port 4000 "$@"
        ;;
    backend)
        set -- uvicorn backend.main:app --host 0.0.0.0 --port 4001 "$@"
        ;;
    ui)
        exec nginx -g 'daemon off;' "$@"
        ;;
    migrations)
        set -- python /app/migrations/run.py "$@"
        ;;
    metrics)
        set -- python -m litellm.proxy.prometheus_metrics_server "$@"
        ;;
    collector)
        set -- python -m litellm.proxy.collector "$@"
        ;;
    *)
        echo "docker-entrypoint.sh: unknown LITELLM_COMPONENT '$component'" >&2
        usage
        exit 64
        ;;
esac

if [ -n "${PROMETHEUS_MULTIPROC_DIR:-}" ]; then
    case "$component" in
        metrics|collector) mkdir -p "$PROMETHEUS_MULTIPROC_DIR" ;;
        *)
            mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
            rm -f "$PROMETHEUS_MULTIPROC_DIR"/*.db
            ;;
    esac
fi

case "${USE_DDTRACE:-}" in
    [Tt][Rr][Uu][Ee])
        export DD_TRACE_OPENAI_ENABLED="False"
        exec ddtrace-run "$@"
        ;;
esac

exec "$@"
