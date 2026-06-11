import Image from '@theme/IdealImage';

# New Relic

## Prerequisite
In order to use LiteLLM with New Relic, you will need to have a New Relic account and [license key](https://docs.newrelic.com/docs/apis/intro-apis/new-relic-api-keys/). If you do not have a New Relic account yet, you can create a [free tier account](https://newrelic.com/pricing/free-tier).

This page covers using New Relic with LiteLLM in proxy mode. You can also use New Relic with the LiteLLM SDK by including the New Relic Python Agent in your application. Please refer to the New Relic AI Monitoring [documentation](https://docs.newrelic.com/docs/ai-monitoring/intro-to-ai-monitoring/).

## Configuration

### Enable New Relic LiteLLM callback

The New Relic LiteLLM extension is implemented as a callback. A common way to enable the callback is via the `config.yaml` file. When configured via `config.yaml`, the `callbacks` list can include several values. As long as `newrelic` is included in the list, the New Relic LiteLLM callback will be invoked. The following example would enable the New Relic callback within LiteLLM.

```yaml
litellm_settings:
  callbacks: ["newrelic"]
```

You can also configure the callback via the LiteLLM administrative UI. If you use this option, please see the
[LiteLLM admin UI documentation](https://docs.litellm.ai/docs/proxy/ui) to access the admin UI and include the New Relic callback within the `Settings` section.

### Required environment variables

The [New Relic Python Agent](https://docs.newrelic.com/docs/apm/agents/python-agent/getting-started/introduction-new-relic-python/) requires configuration to report telemetry to New Relic. The New Relic Python Agent supports defining [configuration](https://docs.newrelic.com/docs/apm/agents/python-agent/configuration/python-agent-configuration/) via both a configuration file and environment variables. With LiteLLM, it is recommended to use environment variables, but both options will work.

The `NEW_RELIC_APP_NAME` environment variable should have a value for the name that you wish the LiteLLM server to appear as in New Relic's UI. The `NEW_RELIC_LICENSE_KEY` environment variable value is a license key for the New Relic account you want the telemetry to be reported to.

```shell
NEW_RELIC_APP_NAME=<app name>
NEW_RELIC_LICENSE_KEY=<license key>
```

## Running LiteLLM with New Relic Python Agent

The [New Relic Python Agent](https://docs.newrelic.com/docs/apm/agents/python-agent/getting-started/introduction-new-relic-python/) is used within applications to report [Application Performance Monitoring (APM)](https://docs.newrelic.com/docs/apm/new-relic-apm/getting-started/introduction-apm/) telemetry to New Relic. By following the steps below, a New Relic customer will receive both the APM telemetry for LiteLLM as well as the LLM messages within [AI Monitoring](https://docs.newrelic.com/docs/ai-monitoring/intro-to-ai-monitoring/) from their LiteLLM server.

### Building a New Relic-enabled container (recommended)

The official LiteLLM containers include the New Relic callback, but do not include the New Relic Python Agent. The easiest way to include the New Relic Python Agent is to create a new container image that layers the agent on top of an existing LiteLLM image. By doing this, you will be able to define the official LiteLLM image version to use as a base.

In order to build a LiteLLM container with the New Relic Python Agent inside of it, you can use the following `Dockerfile`, `entrypoint.sh`, and `supervisord.conf` files. This process will use an official LiteLLM container as a base image, install the New Relic Python Agent, and add new entrypoint and supervisord configuration files. The resulting container will run LiteLLM with the New Relic Python Agent reporting APM telemetry to New Relic. With the callback enabled and the environment variables set above, you will also report the LLM messages to New Relic.

To build the container image, copy the `Dockerfile`, `entrypoint.sh`, and `supervisord.conf` files to a directory. From this directory, you can build an image from the CLI using the following command.

```shell
docker build -f Dockerfile -t litellm-newrelic:local .
```

If you wish to specify the LiteLLM image version to use as a base, pass `--build-arg BASE_IMAGE=…` and/or `--build-arg BASE_TAG=…` to target a different base image or tag similar to the following command.

```shell
BASE_TAG=v1.83.14-stable
docker build \
  --build-arg BASE_IMAGE=docker.litellm.ai/berriai/litellm \
  --build-arg BASE_TAG=${BASE_TAG} \
  -f Dockerfile \
  -t litellm-newrelic:${BASE_TAG} \
  .
```

You may use any docker name for the output image that suits your image naming policy. You will also likely want to push the resulting docker image to your container repository of choice.

#### `Dockerfile`

The Dockerfile defines the layers added on top of an official LiteLLM container. You should pick the version you want to use by setting `BASE_TAG` when you actually build the image. This Dockerfile installs the New Relic Python Agent, then adds new supervisor and entrypoint files that run LiteLLM with the New Relic Python Agent.

```dockerfile
ARG BASE_IMAGE=docker.litellm.ai/berriai/litellm
ARG BASE_TAG=main-stable
FROM ${BASE_IMAGE}:${BASE_TAG}

USER root

# Install New Relic agent (ensurepip bootstraps pip in case base image venv omits it)
RUN python -m ensurepip && python -m pip install --no-cache-dir 'newrelic>=12.1.0,<13'

# Copy New Relic-specific configuration files
COPY supervisord.conf /etc/supervisord_newrelic.conf
COPY entrypoint.sh /app/docker/newrelic/entrypoint.sh
RUN chmod +x /app/docker/newrelic/entrypoint.sh

# Override entrypoint to always use newrelic-admin
ENTRYPOINT ["/app/docker/newrelic/entrypoint.sh"]

LABEL org.opencontainers.image.description="LiteLLM with New Relic APM and AI monitoring"
```

#### `entrypoint.sh`

This `entrypoint.sh` is a copy of LiteLLM's default `docker/prod_entrypoint.sh`, modified to run either supervisord or the `litellm` process wrapped by the New Relic Python Agent.

```sh
#!/bin/sh
# This entry point is a copy of the litellm docker/prod_entrypoint.sh file
# with these changes:
#
#  - Use the New Relic-specific supervisor file
#  - Wrap the litellm command with New Relic Python Agent

if [ "$SEPARATE_HEALTH_APP" = "1" ]; then
    export LITELLM_ARGS="$@"
    export SUPERVISORD_STOPWAITSECS="${SUPERVISORD_STOPWAITSECS:-3600}"
    exec supervisord -c /etc/supervisord_newrelic.conf
fi

exec newrelic-admin run-program litellm "$@"
```

#### `supervisord.conf`

If you use supervisord to run LiteLLM alongside the separate health app, this version will ensure the main LiteLLM process is started with the New Relic Python Agent.

```ini
# This config is a copy of the litellm docker/supervisord.conf with a change to the `main` program
# to wrap the litellm command with the New Relic Python Agent.

[supervisord]
nodaemon=true
loglevel=info
logfile=/tmp/supervisord.log
pidfile=/tmp/supervisord.pid

[group:litellm]
programs=main,health

[program:main]
command=sh -c 'exec newrelic-admin run-program python -m litellm.proxy.proxy_cli --host 0.0.0.0 --port=4000 $LITELLM_ARGS'
autostart=true
autorestart=true
startretries=3
priority=1
exitcodes=0
stopasgroup=true
killasgroup=true
stopwaitsecs=%(ENV_SUPERVISORD_STOPWAITSECS)s
stdout_logfile=/dev/stdout
stderr_logfile=/dev/stderr
stdout_logfile_maxbytes = 0
stderr_logfile_maxbytes = 0
environment=PYTHONUNBUFFERED=true

[program:health]
command=sh -c '[ "$SEPARATE_HEALTH_APP" = "1" ] && exec uvicorn litellm.proxy.health_endpoints.health_app_factory:build_health_app --factory --host 0.0.0.0 --port=${SEPARATE_HEALTH_PORT:-4001} || exit 0'
autostart=true
autorestart=true
startretries=3
priority=2
exitcodes=0
stopasgroup=true
killasgroup=true
stopwaitsecs=%(ENV_SUPERVISORD_STOPWAITSECS)s
stdout_logfile=/dev/stdout
stderr_logfile=/dev/stderr
stdout_logfile_maxbytes = 0
stderr_logfile_maxbytes = 0
environment=PYTHONUNBUFFERED=true

[eventlistener:process_monitor]
command=python -c "from supervisor import childutils; import os, signal; [os.kill(os.getppid(), signal.SIGTERM) for h,p in iter(lambda: childutils.listener.wait(), None) if h['eventname'] in ['PROCESS_STATE_FATAL', 'PROCESS_STATE_EXITED'] and dict([x.split(':') for x in p.split(' ')])['processname'] in ['main', 'health'] or childutils.listener.ok()]"
events=PROCESS_STATE_EXITED,PROCESS_STATE_FATAL
autostart=true
autorestart=true
```

### Running from LiteLLM source

The LiteLLM source uses `uv` to manage dependencies. To run the New Relic integration for LiteLLM, install the New Relic Python Agent locally first. The easiest way is to use `uv` to install it.

```shell
make install-proxy-dev
uv pip install 'newrelic>=12.1.0,<13'
```

You can then run `litellm` locally from source using the New Relic Python Agent with this command. Add any additional options that you might need to the end of the command such as `--config config.yaml` or `--debug`.

```shell
uv run newrelic-admin run-program litellm
```

### Verification

#### LiteLLM container logs

When the New Relic callback initializes, it writes an INFO log message confirming initialization and whether LLM content recording is enabled. If INFO log messages are not visible, set the `LITELLM_LOG=INFO` environment variable to enable them. Look for a message in this form:

```log
New Relic AI Monitoring initialized for app: {app-name}, content recording: {True / False}
```

#### New Relic AI Monitoring

[New Relic AI Monitoring](https://docs.newrelic.com/docs/ai-monitoring/intro-to-ai-monitoring/) can be used to verify the LLM metadata and/or content have been received. From the `AI Responses` view, you should see the LLM requests in the `Responses` table shortly after a request is returned by LiteLLM. If LLM content recording is enabled, the LLM request/response messages will appear in the `Responses` table. Selecting a row on the table will display more details about the LLM request. The trace details may take 2-3 minutes to appear.

## Advanced configuration options

### Disable sending LLM messages to New Relic

You can disable sending LLM messages to New Relic in any of the following ways.

The `config.yaml` file can be used to set a flag that will disable sending LLM messages to New Relic. As you might want LLM messages to be sent elsewhere (logs) but not to New Relic, there is a New Relic-specific configuration value that can be set. Adding the following to your `config.yaml` will prevent LLM messages from being sent to New Relic.

```yaml
litellm_settings:
  callbacks: ["newrelic"]
  newrelic_params:
    turn_off_message_logging: true
```

The New Relic callback also utilizes an environment variable option to disable recording content. This environment variable defaults to `true`. You can turn off recording content messages by setting the following environment variable to `false`.

```shell
NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED=false
```

If you are using the LiteLLM Administrative UI to add the New Relic callback, the form has an option that will accept a boolean value. This boolean value follows the same rules as the environment variable (`true` or blank to record LLM messages, `false` to turn off recording content messages).

### New Relic Agent Configuration

The New Relic Python Agent has a [number of ways](https://docs.newrelic.com/docs/apm/agents/python-agent/configuration/python-agent-configuration/) to accept configuration. As shown above, environment variables can be used to set various optional configuration within the agent.

There is also an agent configuration file that could be used instead of environment variables. If you prefer to use the configuration file, you'll need to ensure the configuration file is accessible. You should also set the following environment variable to point to your configuration file.

```shell
NEW_RELIC_CONFIG_FILE=</path/to/newrelic/configuration_file>
```

### Recommended configuration overrides

The New Relic LiteLLM Extension will send telemetry to New Relic so that the messages appear as part of the New Relic AI Monitoring feature. When using this feature, the following configurations are recommended. These configurations can be set via environment variables or in a configuration file.

```shell
NEW_RELIC_CUSTOM_INSIGHTS_EVENTS_MAX_ATTRIBUTE_VALUE=4095
NEW_RELIC_CUSTOM_INSIGHTS_EVENTS_MAX_SAMPLES_STORED=100000
```

## Support

For support with this integration, contact [New Relic support](https://docs.newrelic.com/docs/new-relic-solutions/solve-common-issues/find-help-get-support/).