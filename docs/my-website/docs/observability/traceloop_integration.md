# Traceloop Tutorial

[Traceloop](https://traceloop.com) is a platform for monitoring and debugging the quality of your LLM outputs.
It provides you with a way to track the performance of your LLM application; rollout changes with confidence; and debug issues in production.
It is based on [OpenTelemetry](https://opentelemetry.io), so it can provide full visibility to your LLM requests, as well vector DB usage, and other infra in your stack.

<Image img={require('../../img/traceloop_dash.png')} />

## Getting Started

First, sign up to get an API key on the [Traceloop dashboard](https://app.traceloop.com).
While Traceloop is still in beta, [ping them](nir@traceloop.com) and mention you're using LiteLLM to get your early access code.

Then, install the Traceloop SDK:

```
pip install traceloop
```

Use just 1 line of code, to instantly log your LLM responses:

```python
litellm.success_callback = ["traceloop"]
```

When running your app, make sure to set the `TRACELOOP_API_KEY` environment variable to your API key.

To get better visualizations on how your code behaves, you may want to annotate specific parts of your LLM chain. See [Traceloop docs on decorators](https://traceloop.com/docs/python-sdk/decorators) for more information.

Here's an example PR for such integration: https://github.com/Codium-ai/pr-agent/pull/244

## Support

For any question or issue with integration you can reach out to the Traceloop team on [Slack](https://join.slack.com/t/traceloopcommunity/shared_invite/zt-1plpfpm6r-zOHKI028VkpcWdobX65C~g) or via [email](mailto:dev@traceloop.com).
