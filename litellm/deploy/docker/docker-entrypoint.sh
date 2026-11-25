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
#   backend     uvicorn litellm.runtime.backend.main:app ARGS    management routes, 0.0.0.0:4001
#   ui          nginx serving the admin UI       port 3000
#   migrations  python litellm/runtime/migrations/run.py         prisma migrate deploy, then exit
#   metrics     python -m litellm.proxy.prometheus_metrics_server ARGS
#   collector   python -m litellm.proxy.collector ARGS
#   --admin-agent  the bundled LiteAdmin Slack agent (litellm-admin-agent --web ARGS)
#
# gateway and backend get their host and port defaults first, so ARGS such as
# --port 8080 override them. With no DATABASE_URL, proxy assembles DATABASE_URL
# and DATABASE_URL_READ_REPLICA from DATABASE_HOST and the other discrete
# variables the way gateway and backend do. An unknown first word is executed
# as-is (docker run <image> sh), through ddtrace-run like every component when
# USE_DDTRACE=true. PgBouncer is not a component: LITELLM_PGBOUNCER_ENABLED=true
# starts it inside proxy and gateway. Components that write Prometheus samples
# start with an empty PROMETHEUS_MULTIPROC_DIR; metrics, collector and raw
# readers keep the workers' files, other raw commands clear stale samples.
set -eu

usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0" >&2
}

component="${LITELLM_COMPONENT:-proxy}"
case "${1:-}" in
    proxy|gateway|backend|ui|migrations|metrics|collector)
        component="$1"
        shift
        ;;
    --admin-agent)
        shift
        export CONNECTION_AUTH_MODE=native
        exec /opt/liteadmin/bin/litellm-admin-agent --web "$@"
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
                *) rm -f "$PROMETHEUS_MULTIPROC_DIR"/*.db "$PROMETHEUS_MULTIPROC_DIR"/litellm_admitted_series_* ;;
            esac
        fi
        component=raw
        ;;
esac

case "$component" in
    raw)
        ;;
    proxy)
        if { [ -z "${DATABASE_URL:-}" ] && [ -n "${DATABASE_HOST:-}" ]; } ||
            { [ -z "${DATABASE_URL_READ_REPLICA:-}" ] && [ -n "${DATABASE_HOST_READ_REPLICA:-}" ]; }; then
            set -- python -c 'import os, sys
from litellm.proxy.db.db_url_settings import (
    DatabaseURLSettings,
    add_missing_query_params,
    connection_params_from_url,
    idle_lifetime_params,
    translate_libpq_ssl_params,
)
if os.environ.get("USE_AWS_KMS") == "True":
    from litellm.secret_managers.aws_secret_manager import decrypt_env_var
    os.environ.update(decrypt_env_var())
settings = DatabaseURLSettings.from_env()
settings.apply_writer_url_to_env()
reader = settings.build_reader_url()
if reader:
    os.environ["DATABASE_URL_READ_REPLICA"] = translate_libpq_ssl_params(
        add_missing_query_params(reader, connection_params_from_url(os.environ.get("DATABASE_URL", "")))
    )
if settings.max_idle_connection_lifetime is not None:
    lifetime = idle_lifetime_params(settings.max_idle_connection_lifetime)
    for name in ("DATABASE_URL", "DATABASE_URL_READ_REPLICA"):
        if os.environ.get(name):
            os.environ[name] = add_missing_query_params(os.environ[name], lifetime)
os.execvp("litellm", ["litellm", *sys.argv[1:]])' "$@"
        else
            set -- litellm "$@"
        fi
        ;;
    gateway)
        set -- python -m gateway.launch --workers "${NUM_WORKERS:-1}" --host 0.0.0.0 --port 4000 "$@"
        ;;
    backend)
        set -- uvicorn litellm.runtime.backend.main:app --host 0.0.0.0 --port 4001 "$@"
        ;;
    ui)
        ui_hook="$(dirname "$0")/ui/docker-entrypoint.d/40-litellm-runtime-config.sh"
        if [ ! -f "$ui_hook" ]; then
            ui_hook="$(dirname "$0")/../../ui/docker-entrypoint.d/40-litellm-runtime-config.sh"
        fi
        sh "$ui_hook"
        exec nginx -g 'daemon off;' "$@"
        ;;
    migrations)
        set -- python -m litellm.runtime.migrations.run "$@"
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
        raw) ;;
        metrics|collector) mkdir -p "$PROMETHEUS_MULTIPROC_DIR" ;;
        *)
            mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
            rm -f "$PROMETHEUS_MULTIPROC_DIR"/*.db "$PROMETHEUS_MULTIPROC_DIR"/litellm_admitted_series_*
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
