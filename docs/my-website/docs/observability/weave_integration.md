# Weave (W&B) (OpenTelemetry)

Weave is available as an OpenTelemetry-backed LiteLLM callback (`weave_otel`).

## Set up

```python
litellm.callbacks = ["weave_otel"]
```

### Environment variables

```shell
WANDB_API_KEY="<your-wandb-api-key>"
WANDB_PROJECT_ID="<entity>/<project>"
WANDB_HOST="<optional-custom-host>"
```

`WANDB_HOST` is optional and defaults to W&B cloud (`trace.wandb.ai`) when omitted.

## Learn more

For details on the W&B endpoint format and tracing docs, see the [OpenTelemetry integration guide](../observability/opentelemetry_integration.md) and the [W&B Weave OTEL docs](https://docs.wandb.ai/weave/guides/tracing/otel).
