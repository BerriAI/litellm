import Image from '@theme/IdealImage';

# What is New Relic?

# New Relic Extension in LiteLLM

## Enable New Relic Python Agent instrumentation

The LiteLLM proxy has an optional configuration to enable instrumentation with the New Relic Python Agent. In order for the New Relic extension to work within the LiteLLM proxy, the New Relic Python Agent must instrument the process. This will result in the LiteLLM proxy reporting telemetry about the proxy to New Relic.

In order to enable the New Relic Python Agent with the LiteLLM proxy, include the following environment variable when starting the proxy.

```shell
USE_NEWRELIC=true
```

## Enable New Relic LiteLLM Extension

The New Relic LiteLLM extension is implemented as a callback. A common way to enable the callback is via the config.yaml definition. In order to enable the New Relic extension, add a callback configuration similar to the following in your config.yaml. The callbacks can include a number of different extensions. As long as ”newrelic” is included in the list, the New Relic LiteLLM extension will be invoked.

```yaml
litellm_settings:
  callbacks: ["newrelic"]
```

## Required environment variables

In order for the New Relic Python Agent to report telemetry to New Relic, there are a few environment variables that should be set.

The NEW_RELIC_APP_NAME should have a value for the name that you wish the LiteLLM server to appear as in New Relic’s UI. The NEW_RELIC_LICENSE_KEY value is a license key for the New Relic account you want the telemetry to be reported to.

The USE_NEWRELIC is required in order to enable the New Relic Python Agent with the LiteLLM proxy. This must be sent in order for the New Relic Python Agent to report telemetry to New Relic.

```shell
NEW_RELIC_APP_NAME=<app name>
NEW_RELIC_LICENSE_KEY=<license key>

USE_NEWRELIC=true
```

## Optional Configuration

### Disable sending LLM messages to New Relic

There are two options to disable sending LLM messages to New Relic. Using either of these options will disable sending LLM messages to New Relic; you do not need to set both.

The config.yaml file can be used to set a flag that will disable sending LLM messages to New Relic. As you might want LLM messages to be sent elsewhere (logs) but not to New Relic, there is a New Relic specific configuration value that can be set. Adding the following to your config.yaml will prevent LLM messages from being sent to New Relic.

```yaml
litellm_settings:
  callbacks: ["newrelic"]
  newrelic_params:
    turn_off_message_logging: true
```

The other option is to use an environment variable to disable sending LLM messages to New Relic. This can be achieved with the following environment variable.

```shell
NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED=false
```

### New Relic Agent Configuration

The New Relic Python Agent has a (number of ways)[https://docs.newrelic.com/docs/apm/agents/python-agent/configuration/python-agent-configuration/] to accept configuration. As shown above, environment variables can be used to set various optional configuration within the agent.

There is also an agent configuration file that could be used instead of environment variables. If you prefer to use the configuration file, you’ll need to ensure the configuration file is accessible. You should also set the following environment variable to point to your configuration file.

```shell
NEW_RELIC_CONFIG_FILE=</path/to/newrelic/configuration_file>
```

### Recommended configuration overrides

The New Relic LiteLLM Extension will send telemetry to New Relic so that the messages appear as part of the New Relic AI Monitoring feature. When using this feature, there are some configurations that are recommended to be set. This configurations can be set via environment variables or in a configuration file. The following environment variables are recommended.

```shell
NEW_RELIC_CUSTOM_INSIGHTS_EVENTS_MAX_ATTRIBUTE_VALUE=4095
NEW_RELIC_EVENT_HARVEST_CONFIG_HARVEST_LIMITS_CUSTOM_EVENT_DATA=100000
```
