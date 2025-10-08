import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Helicone - OSS LLM Observability Platform

:::tip

This is community maintained. Please make an issue if you run into a bug:
https://github.com/BerriAI/litellm

:::

[Helicone](https://helicone.ai/) is an open source observability platform that proxies your LLM requests and provides key insights into your usage, spend, latency and more.

## Quick Start

<Tabs>
<TabItem value="sdk" label="Python SDK">

Use just 1 line of code to instantly log your responses **across all providers** with Helicone:

```python
import os
from litellm import completion

## Set env variables
os.environ["HELICONE_API_KEY"] = "your-helicone-key"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Set callbacks
litellm.success_callback = ["helicone"]

# OpenAI call
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi 👋 - I'm OpenAI"}],
)

print(response)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

Add Helicone to your LiteLLM proxy configuration:

```yaml title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY

# Add Helicone callback
litellm_settings:
  success_callback: ["helicone"]
  
# Set Helicone API key
environment_variables:
  HELICONE_API_KEY: "your-helicone-key"
```

Start the proxy:
```bash
litellm --config config.yaml
```

</TabItem>
</Tabs>

## Integration Methods

There are two main approaches to integrate Helicone with LiteLLM:

1. **Callbacks**: Log to Helicone while using any provider
2. **Proxy Mode**: Use Helicone as a proxy for advanced features

### Supported LLM Providers

Helicone can log requests across [various LLM providers](https://docs.helicone.ai/getting-started/quick-start), including:

- OpenAI
- Azure
- Anthropic
- Gemini
- Groq
- Cohere
- Replicate
- And more

## Method 1: Using Callbacks

Log requests to Helicone while using any LLM provider directly.

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
import os
import litellm
from litellm import completion

## Set env variables
os.environ["HELICONE_API_KEY"] = "your-helicone-key"
os.environ["OPENAI_API_KEY"] = "your-openai-key"
# os.environ["HELICONE_API_BASE"] = "" # [OPTIONAL] defaults to `https://api.helicone.ai`

# Set callbacks
litellm.success_callback = ["helicone"]

# OpenAI call
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi 👋 - I'm OpenAI"}],
)

print(response)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```yaml title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-3
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

# Add Helicone logging
litellm_settings:
  success_callback: ["helicone"]
  
# Environment variables
environment_variables:
  HELICONE_API_KEY: "your-helicone-key"
  OPENAI_API_KEY: "your-openai-key"
  ANTHROPIC_API_KEY: "your-anthropic-key"
```

Start the proxy:
```bash
litellm --config config.yaml
```

Make requests to your proxy:
```python
import openai

client = openai.OpenAI(
    api_key="anything",  # proxy doesn't require real API key
    base_url="http://localhost:4000"
)

response = client.chat.completions.create(
    model="gpt-4",  # This gets logged to Helicone
    messages=[{"role": "user", "content": "Hello!"}]
)
```

</TabItem>
</Tabs>

## Method 2: Using Helicone as a Proxy

Helicone's proxy provides [advanced functionality](https://docs.helicone.ai/getting-started/proxy-vs-async) like caching, rate limiting, LLM security through [PromptArmor](https://promptarmor.com/) and more.

<Tabs>
<TabItem value="sdk" label="Python SDK">

Set Helicone as your base URL and pass authentication headers:

```python
import os
import litellm
from litellm import completion

# Configure LiteLLM to use Helicone proxy
litellm.api_base = "https://oai.hconeai.com/v1"
litellm.headers = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",
}

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-key"

response = completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "How does a court case get to the Supreme Court?"}]
)

print(response)
```

### Advanced Usage

You can add custom metadata and properties to your requests using Helicone headers. Here are some examples:

```python
litellm.metadata = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",  # Authenticate to send requests to Helicone API
    "Helicone-User-Id": "user-abc",  # Specify the user making the request
    "Helicone-Property-App": "web",  # Custom property to add additional information
    "Helicone-Property-Custom": "any-value",  # Add any custom property
    "Helicone-Prompt-Id": "prompt-supreme-court",  # Assign an ID to associate this prompt with future versions
    "Helicone-Cache-Enabled": "true",  # Enable caching of responses
    "Cache-Control": "max-age=3600",  # Set cache limit to 1 hour
    "Helicone-RateLimit-Policy": "10;w=60;s=user",  # Set rate limit policy
    "Helicone-Retry-Enabled": "true",  # Enable retry mechanism
    "helicone-retry-num": "3",  # Set number of retries
    "helicone-retry-factor": "2",  # Set exponential backoff factor
    "Helicone-Model-Override": "gpt-3.5-turbo-0613",  # Override the model used for cost calculation
    "Helicone-Session-Id": "session-abc-123",  # Set session ID for tracking
    "Helicone-Session-Path": "parent-trace/child-trace",  # Set session path for hierarchical tracking
    "Helicone-Omit-Response": "false",  # Include response in logging (default behavior)
    "Helicone-Omit-Request": "false",  # Include request in logging (default behavior)
    "Helicone-LLM-Security-Enabled": "true",  # Enable LLM security features
    "Helicone-Moderations-Enabled": "true",  # Enable content moderation
    "Helicone-Fallbacks": '["gpt-3.5-turbo", "gpt-4"]',  # Set fallback models
}
```

### Caching and Rate Limiting

Enable caching and set up rate limiting policies:

```python
litellm.metadata = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",  # Authenticate to send requests to Helicone API
    "Helicone-Cache-Enabled": "true",  # Enable caching of responses
    "Cache-Control": "max-age=3600",  # Set cache limit to 1 hour
    "Helicone-RateLimit-Policy": "100;w=3600;s=user",  # Set rate limit policy
}
```

</TabItem>
</Tabs>

## Session Tracking and Tracing

Track multi-step and agentic LLM interactions using session IDs and paths:

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
import litellm

litellm.api_base = "https://oai.hconeai.com/v1"
litellm.metadata = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",
    "Helicone-Session-Id": "session-abc-123",
    "Helicone-Session-Path": "parent-trace/child-trace",
}

response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Start a conversation"}]
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```python
import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://localhost:4000"
)

# First request in session
response1 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "Helicone-Session-Id": "session-abc-123",
        "Helicone-Session-Path": "conversation/greeting"
    }
)

# Follow-up request in same session
response2 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me more"}],
    extra_headers={
        "Helicone-Session-Id": "session-abc-123",
        "Helicone-Session-Path": "conversation/follow-up"
    }
)
```

</TabItem>
</Tabs>

- `Helicone-Session-Id`: Unique identifier for the session to group related requests
- `Helicone-Session-Path`: Hierarchical path to represent parent/child traces (e.g., "parent/child")

## Retry and Fallback Mechanisms

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
import litellm

litellm.api_base = "https://oai.hconeai.com/v1"
litellm.metadata = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",
    "Helicone-Retry-Enabled": "true",
    "helicone-retry-num": "3",
    "helicone-retry-factor": "2",  # Exponential backoff
    "Helicone-Fallbacks": '["gpt-3.5-turbo", "gpt-4"]',
}

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```yaml title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY
      api_base: "https://oai.hconeai.com/v1"

default_litellm_params:
  headers:
    Helicone-Auth: "Bearer ${HELICONE_API_KEY}"
    Helicone-Retry-Enabled: "true"
    helicone-retry-num: "3"
    helicone-retry-factor: "2"
    Helicone-Fallbacks: '["gpt-3.5-turbo", "gpt-4"]'

environment_variables:
  HELICONE_API_KEY: "your-helicone-key"
  OPENAI_API_KEY: "your-openai-key"
```

</TabItem>
</Tabs>

> **Supported Headers** - For a full list of supported Helicone headers and their descriptions, please refer to the [Helicone documentation](https://docs.helicone.ai/getting-started/quick-start).
> By utilizing these headers and metadata options, you can gain deeper insights into your LLM usage, optimize performance, and better manage your AI workflows with Helicone and LiteLLM.
