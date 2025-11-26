import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenTelemetry - Tracing LLMs with any observability tool

OpenTelemetry is a CNCF standard for observability. It connects to any observability tool, such as Jaeger, Zipkin, Datadog, New Relic, Traceloop and others.

<Image img={require('../../img/traceloop_dash.png')} />

:::note Change in v1.81.0

From v1.81.0, the request/response will be set as attributes on the parent "Received Proxy Server Request" span by default. This allows you to see the request/response in the parent span in your observability tool.

To use the older behavior with nested "litellm_request" spans, set the following environment variable:

```shell
USE_OTEL_LITELLM_REQUEST_SPAN=true
```

:::

## Getting Started

Install the OpenTelemetry SDK:

```
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

Set the environment variables (different providers may require different variables):


<Tabs>

<TabItem value="traceloop" label="Log to Traceloop Cloud">

```shell
OTEL_EXPORTER="otlp_http"
OTEL_ENDPOINT="https://api.traceloop.com"
OTEL_HEADERS="Authorization=Bearer%20<your-api-key>"
```

</TabItem>

<TabItem value="otel-col" label="Log to OTEL HTTP Collector">

```shell
OTEL_EXPORTER_OTLP_ENDPOINT="http://0.0.0.0:4318"
OTEL_EXPORTER_OTLP_PROTOCOL=http/json
OTEL_EXPORTER_OTLP_HEADERS="api-key=key,other-config-value=value"
```

</TabItem>

<TabItem value="otel-col-grpc" label="Log to OTEL GRPC Collector">

```shell
OTEL_EXPORTER_OTLP_ENDPOINT="http://0.0.0.0:4318"
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_HEADERS="api-key=key,other-config-value=value"
```

</TabItem>

<TabItem value="laminar" label="Log to Laminar">

```shell
OTEL_EXPORTER="otlp_grpc"
OTEL_ENDPOINT="https://api.lmnr.ai:8443"
OTEL_HEADERS="authorization=Bearer <project-api-key>"
```

</TabItem>

</Tabs>

Use just 1 line of code, to instantly log your LLM responses **across all providers** with OpenTelemetry:

```python
litellm.callbacks = ["otel"]
```

## Redacting Messages, Response Content from OpenTelemetry Logging

### Redact Messages and Responses from all OpenTelemetry Logging

Set `litellm.turn_off_message_logging=True` This will prevent the messages and responses from being logged to OpenTelemetry, but request metadata will still be logged.

### Redact Messages and Responses from specific OpenTelemetry Logging

In the metadata typically passed for text completion or embedding calls you can set specific keys to mask the messages and responses for this call.

Setting `mask_input` to `True` will mask the input from being logged for this call

Setting `mask_output` to `True` will make the output from being logged for this call.

Be aware that if you are continuing an existing trace, and you set `update_trace_keys` to include either `input` or `output` and you set the corresponding `mask_input` or `mask_output`, then that trace will have its existing input and/or output replaced with a redacted message.

## Support

For any question or issue with the integration you can reach out to the OpenLLMetry maintainers on [Slack](https://traceloop.com/slack) or via [email](mailto:dev@traceloop.com).

## Troubleshooting

### Trace LiteLLM Proxy user/key/org/team information on failed requests

LiteLLM emits the user_api_key_metadata
- key hash
- key_alias
- org_id
- user_id
- team_id

for successful + failed requests

click under `litellm_request` in the trace

<Image img={require('../../img/otel_debug_trace.png')} />

### Not seeing traces land on Integration

If you don't see traces landing on your integration, set `OTEL_DEBUG="True"` in your LiteLLM environment and try again.

```shell
export OTEL_DEBUG="True"
```

This will emit any logging issues to the console.