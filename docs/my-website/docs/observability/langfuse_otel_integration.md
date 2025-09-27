import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import Image from '@theme/IdealImage';

# 🪢 Langfuse OpenTelemetry Integration

The Langfuse OpenTelemetry integration allows you to send LiteLLM traces and observability data to Langfuse using the OpenTelemetry protocol. This provides a standardized way to collect and analyze your LLM usage data.

<Image img={require('../../img/langfuse_otel.png')} />

## Features

- Automatic trace collection for all LiteLLM requests
- Support for Langfuse Cloud (EU and US regions)
- Support for self-hosted Langfuse instances
- Custom endpoint configuration
- Secure authentication using Basic Auth
- Consistent attribute mapping with other OTEL integrations

## Prerequisites

1. **Langfuse Account**: Sign up at [Langfuse Cloud](https://cloud.langfuse.com) or set up a self-hosted instance
2. **API Keys**: Get your public and secret keys from your Langfuse project settings
3. **Dependencies**: Install required packages:
   ```bash
   pip install litellm opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
   ```

## Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `LANGFUSE_PUBLIC_KEY` | Yes | Your Langfuse public key | `pk-lf-...` |
| `LANGFUSE_SECRET_KEY` | Yes | Your Langfuse secret key | `sk-lf-...` |
| `LANGFUSE_OTEL_HOST` | No | OTEL endpoint host | `https://otel.my-langfuse.com` |

### Endpoint Resolution

The integration automatically constructs the OTEL endpoint from `LANGFUSE_OTEL_HOST`
- **Default (US)**: `https://us.cloud.langfuse.com/api/public/otel`
- **EU Region**: `https://cloud.langfuse.com/api/public/otel`
- **Self-hosted**: `{LANGFUSE_OTEL_HOST}/api/public/otel`

## Usage

### Basic Setup

```python
import os
import litellm

# Set your Langfuse credentials
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."

# Enable Langfuse OTEL integration
litellm.callbacks = ["langfuse_otel"]

# Make LLM requests as usual
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Advanced Configuration

```python
import os
import litellm

# Set your Langfuse credentials
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."

# Use EU region
os.environ["LANGFUSE_OTEL_HOST"] = "https://cloud.langfuse.com"  # EU region
# os.environ["LANGFUSE_OTEL_HOST"] = "https://otel.my-langfuse.company.com"  # custom OTEL endpoint

# Or use self-hosted instance
# os.environ["LANGFUSE_OTEL_HOST"] = "https://my-langfuse.company.com"

litellm.callbacks = ["langfuse_otel"]
```

### Manual OTEL Configuration

If you need direct control over the OpenTelemetry configuration:

```python
import os
import base64
import litellm

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..." 
os.environ["LANGFUSE_OTEL_HOST"] = "https://cloud.langfuse.com" # EU region
# os.environ["LANGFUSE_OTEL_HOST"] = "https://us.cloud.langfuse.com" # US region
# os.environ["LANGFUSE_OTEL_HOST"] = "https://otel.my-langfuse.company.com" # custom OTEL endpoint

LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
).decode()

host = os.environ.get("LANGFUSE_OTEL_HOST")
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = host + "/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

litellm.callbacks = ["langfuse_otel"]
```

### With LiteLLM Proxy

Add the integration to your proxy configuration:

1. Add the credentials to your environment variables

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_OTEL_HOST="https://us.cloud.langfuse.com"  # Default US region
# export LANGFUSE_OTEL_HOST="https://otel.my-langfuse.company.com"  # custom OTEL endpoint
```

2. Setup config.yaml

```yaml
# config.yaml
litellm_settings:
  callbacks: ["langfuse_otel"]
```

3. Run the proxy

```bash
litellm --config /path/to/config.yaml
```

## Data Collected

The integration automatically collects the following data:

- **Request Details**: Model, messages, parameters (temperature, max_tokens, etc.)
- **Response Details**: Generated content, token usage, finish reason
- **Timing Information**: Request duration, time to first token
- **Metadata**: User ID, session ID, custom tags (if provided)
- **Error Information**: Exception details and stack traces (if errors occur)

## Metadata Support

All metadata fields available in the vanilla Langfuse integration are now **fully supported** when you use the OTEL integration.

- Any key you pass in the `metadata` dictionary (`generation_name`, `trace_id`, `session_id`, `tags`, and the rest) is exported as an OpenTelemetry span attribute.
- Attribute names are prefixed with `langfuse.` so you can filter or search for them easily in your observability backend.
  Examples: `langfuse.generation.name`, `langfuse.trace.id`, `langfuse.trace.session_id`.

### Passing Metadata – Example

```python
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    metadata={
        "generation_name": "welcome-message",
        "trace_id": "trace-123",
        "session_id": "sess-42",
        "tags": ["prod", "beta-user"]
    }
)
```

The resulting span will contain attributes similar to:

```
langfuse.generation.name   = "welcome-message"
langfuse.trace.id          = "trace-123"
langfuse.trace.session_id  = "sess-42"
langfuse.trace.tags        = ["prod", "beta-user"]
```

Use the **Langfuse UI** (Traces tab) to search, filter and analyse spans that contain the `langfuse.*` attributes.
The OTEL exporter in this integration sends data directly to Langfuse’s OTLP HTTP endpoint; it is **not** intended for Grafana, Honeycomb, Datadog, or other generic OTEL back-ends.

## Authentication

The integration uses HTTP Basic Authentication with your Langfuse public and secret keys:

```
Authorization: Basic <base64(public_key:secret_key)>
```

This is automatically handled by the integration - you just need to provide the keys via environment variables.

## Troubleshooting

### Common Issues

1. **Missing Credentials Error**
   ```
   ValueError: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set
   ```
   **Solution**: Ensure both environment variables are set with valid keys.

2. **Connection Issues**
   - Check your internet connection
   - Verify the endpoint URL is correct
   - For self-hosted instances, ensure the `/api/public/otel` endpoint is accessible

3. **Authentication Errors**
   - Verify your public and secret keys are correct
   - Check that the keys belong to the same Langfuse project
   - Ensure the keys have the necessary permissions

### Debug Mode

Enable verbose logging to see detailed information:

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm._turn_on_debug()
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
export LITELLM_LOG="DEBUG"
```

</TabItem>
</Tabs>

This will show:
- Endpoint resolution logic
- Authentication header creation
- OTEL trace submission details

## Related Links

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse OpenTelemetry Guide](https://langfuse.com/docs/integrations/opentelemetry)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
- [LiteLLM Observability](https://docs.litellm.ai/docs/observability/) 