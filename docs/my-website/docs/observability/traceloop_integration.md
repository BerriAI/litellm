import Image from '@theme/IdealImage';

# Traceloop (OpenLLMetry) - Tracing LLMs with OpenTelemetry

[Traceloop](https://traceloop.com) is a platform for monitoring and debugging the quality of your LLM outputs.
It provides you with a way to track the performance of your LLM application; rollout changes with confidence; and debug issues in production.
It is based on [OpenTelemetry](https://opentelemetry.io), so it can provide full visibility to your LLM requests, as well vector DB usage, and other infra in your stack.

<Image img={require('../../img/traceloop_dash.png')} />

## Getting Started

Install the Traceloop SDK:

```
pip install traceloop-sdk
```

Use just 2 lines of code, to instantly log your LLM responses with OpenTelemetry:

```python
Traceloop.init(app_name=<YOUR APP NAME>, disable_batch=True)
litellm.success_callback = ["traceloop"]
```

Make sure to properly set a destination to your traces. See [OpenLLMetry docs](https://www.traceloop.com/docs/openllmetry/integrations/introduction) for options.

To get better visualizations on how your code behaves, you may want to annotate specific parts of your LLM chain. See [Traceloop docs on decorators](https://traceloop.com/docs/python-sdk/decorators) for more information.

## Exporting traces to other systems (e.g. Datadog, New Relic, and others)

Since OpenLLMetry uses OpenTelemetry to send data, you can easily export your traces to other systems, such as Datadog, New Relic, and others. See [OpenLLMetry docs on exporters](https://www.traceloop.com/docs/openllmetry/integrations/introduction) for more information.

## Support

For any question or issue with integration you can reach out to the Traceloop team on [Slack](https://traceloop.com/slack) or via [email](mailto:dev@traceloop.com).
