# Helicone - OSS LLM Observability Platform

:::tip

This is community maintained. Please make an issue if you run into a bug:
https://github.com/BerriAI/litellm

:::

[Helicone](https://helicone.ai/) is an open source observability platform that proxies your LLM requests and provides key insights into your usage, spend, latency and more.

## Using Helicone with LiteLLM

LiteLLM provides `success_callbacks` and `failure_callbacks`, allowing you to easily log data to Helicone based on the status of your responses.

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

### Integration Methods

There are two main approaches to integrate Helicone with LiteLLM:

1. Using callbacks
2. Using Helicone as a proxy

Let's explore each method in detail.

### Approach 1: Use Callbacks

Use just 1 line of code to instantly log your responses **across all providers** with Helicone:

```python
litellm.success_callback = ["helicone"]
```

Complete Code

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

### Approach 2: Use Helicone as a proxy

Helicone's proxy provides [advanced functionality](https://docs.helicone.ai/getting-started/proxy-vs-async) like caching, rate limiting, LLM security through [PromptArmor](https://promptarmor.com/) and more.

To use Helicone as a proxy for your LLM requests:

1. Set Helicone as your base URL via: litellm.api_base
2. Pass in Helicone request headers via: litellm.metadata

Complete Code:

```python
import os
import litellm
from litellm import completion

litellm.api_base = "https://oai.hconeai.com/v1"
litellm.headers = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",  # Authenticate to send requests to Helicone API
}

response = litellm.completion(
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

### Session Tracking and Tracing

Track multi-step and agentic LLM interactions using session IDs and paths:

```python
litellm.metadata = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",  # Authenticate to send requests to Helicone API
    "Helicone-Session-Id": "session-abc-123",  # The session ID you want to track
    "Helicone-Session-Path": "parent-trace/child-trace",  # The path of the session
}
```

- `Helicone-Session-Id`: Use this to specify the unique identifier for the session you want to track. This allows you to group related requests together.
- `Helicone-Session-Path`: This header defines the path of the session, allowing you to represent parent and child traces. For example, "parent/child" represents a child trace of a parent trace.

By using these two headers, you can effectively group and visualize multi-step LLM interactions, gaining insights into complex AI workflows.

### Retry and Fallback Mechanisms

Set up retry mechanisms and fallback options:

```python
litellm.metadata = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",  # Authenticate to send requests to Helicone API
    "Helicone-Retry-Enabled": "true",  # Enable retry mechanism
    "helicone-retry-num": "3",  # Set number of retries
    "helicone-retry-factor": "2",  # Set exponential backoff factor
    "Helicone-Fallbacks": '["gpt-3.5-turbo", "gpt-4"]',  # Set fallback models
}
```

> **Supported Headers** - For a full list of supported Helicone headers and their descriptions, please refer to the [Helicone documentation](https://docs.helicone.ai/getting-started/quick-start).
> By utilizing these headers and metadata options, you can gain deeper insights into your LLM usage, optimize performance, and better manage your AI workflows with Helicone and LiteLLM.
