---
sidebar_label: Levo AI
---

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Levo AI

<div className="levo-logo-container" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
  <div className="levo-logo-light">
    <Image img={require('../../img/levo_logo.png')} />
  </div>
  <div className="levo-logo-dark">
    <Image img={require('../../img/levo_logo_dark.png')} />
  </div>
</div>

[Levo](https://levo.ai/) is an AI observability and compliance platform that provides comprehensive monitoring, analysis, and compliance tracking for LLM applications.

## Quick Start

Send all your LLM requests and responses to Levo for monitoring and analysis using LiteLLM's built-in OpenTelemetry support.

### What You'll Get

- **Complete visibility** into all LLM API calls across all providers
- **Request and response data** including prompts, completions, and metadata
- **Usage and cost tracking** with token counts and cost breakdowns
- **Error monitoring** and performance metrics
- **Compliance tracking** for audit and governance

### Setup Steps

**1. Install OpenTelemetry dependencies:**

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

**2. Enable OpenTelemetry in your LiteLLM config:**

Add to your `litellm_config.yaml`:

```yaml
litellm_settings:
  callbacks: ["otel"]
  turn_off_message_logging: false  # Required to capture request/response data
```

**3. Configure environment variables:**

[Contact Levo support](mailto:support@levo.ai) to get your OpenTelemetry collector endpoint URL and your Levo organization ID.

Set these environment variables:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="<your-levo-collector-url>"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_SERVICE_NAME="litellm-proxy"
export OTEL_EXPORTER_OTLP_HEADERS="x-levo-organization-id=<your-levo-org-id>,x-levo-workspace-id=<your-workspace-id>"
```

**4. Start LiteLLM:**

```bash
litellm --config config.yaml
```

**5. Make requests - they'll automatically be sent to Levo!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "Hello, this is a test message"
        }
    ]
    }'
```

### Customizing Service Name

You can customize how your service appears in Levo:

```bash
export OTEL_SERVICE_NAME="my-litellm-service"
export OTEL_RESOURCE_ATTRIBUTES="service.name=my-litellm-service"
```

## What Data is Captured

| Feature | Details |
|---------|---------|
| **What is logged** | OpenTelemetry Trace Data (OTLP format) |
| **Events** | Success + Failure |
| **Format** | OTLP (OpenTelemetry Protocol) - see [OpenTelemetry Integration](./opentelemetry_integration) for details |

## Troubleshooting

### Not seeing traces in Levo?

1. **Verify OpenTelemetry is enabled**: Check LiteLLM startup logs for `initializing callbacks=['otel']`

2. **Check environment variables**: Ensure `OTEL_EXPORTER_OTLP_ENDPOINT` is set correctly:
   ```bash
   echo $OTEL_EXPORTER_OTLP_ENDPOINT
   ```

3. **Verify collector connectivity**: Test if your collector is reachable:
   ```bash
   curl <your-collector-url>/health
   ```

4. **Enable debug logging**:
   ```bash
   export OTEL_LOG_LEVEL="DEBUG"
   export OTEL_DEBUG="True"
   export LITELLM_LOG="DEBUG"
   ```

5. **Wait for async export**: OTLP sends traces asynchronously. Wait 10-15 seconds after making requests before checking Levo.

## Additional Resources

- [Levo Documentation](https://docs.levo.ai)
- [LiteLLM OpenTelemetry Integration](./opentelemetry_integration)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)

## Need Help?

For issues or questions about the Levo integration with LiteLLM, please [contact Levo support](mailto:support@levo.ai) or open an issue on the [LiteLLM GitHub repository](https://github.com/BerriAI/litellm/issues).
