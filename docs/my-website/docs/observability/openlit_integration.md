import Image from '@theme/IdealImage';

# OpenLIT - OpenTelemetry-native LLM Observability

[OpenLIT](https://github.com/openlit/openlit) is an OpenTelemetry-native Auto instrumentation library for monitoring LLM Applications, facilitating the integration of observability into your GenAI-driven projects using just a single line of code.

![OpenLIT Connections Banner](https://github.com/openlit/.github/blob/main/profile/assets/github-readme-connections-banner.png?raw=true)

## Getting Started

### Step 1: Install OpenLIT

Open your command line or terminal and run:

```bash
pip install openlit
```

### Step 2: Initialize OpenLIT in your Application
Integrating OpenLIT into LLM applications is straightforward. Start monitoring for your LLM Application with just **two lines of code**: 

```python
import openlit

openlit.init()
```

To forward telemetry data to an HTTP OTLP endpoint, such as the OpenTelemetry Collector, set the `otlp_endpoint` parameter with the desired endpoint. Alternatively, you can configure the endpoint by setting the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable as recommended in the OpenTelemetry documentation.

> 💡 Info: If you don't provide `otlp_endpoint` function argument or set the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable, OpenLIT directs the trace directly to your console, which can be useful during development.

To send telemetry to OpenTelemetry backends requiring authentication, set the `otlp_headers` parameter with its desired value. Alternatively, you can configure the endpoint by setting the `OTEL_EXPORTER_OTLP_HEADERS` environment variable as recommended in the OpenTelemetry documentation.

### Step 3: Visualize and Optimize!

![](https://github.com/openlit/.github/blob/main/profile/assets/openlit-client-1.png?raw=true)

With the LLM Observability data now being collected by OpenLIT, the next step is to visualize and analyze this data to get insights into your LLM application’s performance, behavior, and identify areas of improvement.

To begin exploring your LLM Application's performance data within the OpenLIT UI, please see the [Quickstart Guide](https://docs.openlit.io/latest/quickstart).

If you want to integrate and send metrics and traces to your existing observability tools like Promethues+Jaeger, Grafana or more, refer to the [Official Documentation for OpenLIT Connections](https://docs.openlit.io/latest/connections/intro) for detailed instructions.


## Support

For any question or issue with integration you can reach out to the OpenLIT team on [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ) or via [email](mailto:contact@openlit.io).
