#!/bin/sh
# ⚠️ KEEP IN SYNC: Changes to docker/prod_entrypoint.sh logic should be reviewed here

if [ "$SEPARATE_HEALTH_APP" = "1" ]; then
    export LITELLM_ARGS="$@"
    export SUPERVISORD_STOPWAITSECS="${SUPERVISORD_STOPWAITSECS:-3600}"
    exec supervisord -c /etc/supervisord_newrelic.conf
fi

exec newrelic-admin run-program litellm "$@"
