import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Dynamic Callback Management

:::info

This is an enterprise feature.

[Get started with LiteLLM Enterprise](https://www.litellm.ai/enterprise)

:::

LiteLLM's dynamic callback management enables teams to control logging behavior on a per-request basis without requiring central infrastructure changes. This is essential for organizations managing large-scale service ecosystems where:

- **Teams manage their own compliance** - Services can handle sensitive data appropriately without central oversight
- **Decentralized responsibility** - Each team controls their data handling while using shared infrastructure

You can disable callbacks by passing the `x-litellm-disable-callbacks` header with your requests, giving teams granular control over where their data is logged.

## Quick Start

```bash
# Disable a single callback
curl -H "x-litellm-disable-callbacks: langfuse" ...

# Disable multiple callbacks
curl -H "x-litellm-disable-callbacks: langfuse,datadog" ...
```

## 1. View Active Logging Callbacks

Before disabling callbacks, you can view all currently enabled callbacks on your proxy.

### Request

```bash
curl --location 'http://0.0.0.0:4000/callbacks/list' \
    --header 'Authorization: Bearer sk-1234'
```

### Response

```json
{
    "callbacks": [
        "langfuse",
        "datadog", 
        "prometheus",
        "slack_alerting"
    ]
}
```

## 2. Disable a Single Callback

Use the `x-litellm-disable-callbacks` header to disable specific callbacks for individual requests.

<Tabs>
<TabItem value="Curl" label="Curl Request">

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'x-litellm-disable-callbacks: langfuse' \
    --data '{
    "model": "claude-sonnet-4-20250514",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
```

</TabItem>
<TabItem value="OpenAI" label="OpenAI Python SDK">

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[
        {
            "role": "user",
            "content": "what llm are you"
        }
    ],
    extra_headers={
        "x-litellm-disable-callbacks": "langfuse"
    }
)

print(response)
```

</TabItem>
</Tabs>

## 3. Disable Multiple Callbacks

You can disable multiple callbacks by providing a comma-separated list in the header.

<Tabs>
<TabItem value="Curl" label="Curl Request">

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'x-litellm-disable-callbacks: langfuse,datadog,prometheus' \
    --data '{
    "model": "claude-sonnet-4-20250514",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
```

</TabItem>
<TabItem value="OpenAI" label="OpenAI Python SDK">

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[
        {
            "role": "user",
            "content": "what llm are you"
        }
    ],
    extra_headers={
        "x-litellm-disable-callbacks": "langfuse,datadog,prometheus"
    }
)

print(response)
```

</TabItem>
</Tabs>
