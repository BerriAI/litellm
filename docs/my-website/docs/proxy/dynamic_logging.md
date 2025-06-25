import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Dynamic Callback Management

:::info

✨ This is an enterprise feature.

[Get started with LiteLLM Enterprise](https://www.litellm.ai/enterprise)

:::

LiteLLM's dynamic callback management enables teams to control logging behavior on a per-request basis without requiring central infrastructure changes. This is essential for organizations managing large-scale service ecosystems where:

- **Teams manage their own compliance** - Services can handle sensitive data appropriately without central oversight
- **Decentralized responsibility** - Each team controls their data handling while using shared infrastructure

You can disable callbacks by passing the `x-litellm-disable-callbacks` header with your requests, giving teams granular control over where their data is logged.

## Getting Started: List and Disable Callbacks

Managing callbacks is a two-step process:

1. **First, list your active callbacks** to see what's currently enabled
2. **Then, disable specific callbacks** as needed for your requests



## 1. List Active Callbacks

Start by viewing all currently enabled callbacks on your proxy to see what's available to disable.

#### Request

```bash
curl -X 'GET' \
  'http://localhost:4000/callbacks/list' \
  -H 'accept: application/json' \
  -H 'x-litellm-api-key: sk-1234'
```

#### Response

```json
{
  "success": [
    "deployment_callback_on_success",
    "sync_deployment_callback_on_success"
  ],
  "failure": [
    "async_deployment_callback_on_failure",
    "deployment_callback_on_failure"
  ],
  "success_and_failure": [
    "langfuse",
    "datadog"
  ]
}
```

#### Response Fields

The response contains three arrays that categorize your active callbacks:
- **`success`** - Callbacks that only execute when requests complete successfully. These callbacks receive data from successful LLM responses.
- **`failure`** - Callbacks that only execute when requests fail or encounter errors. These callbacks receive error information and failed request data.
- **`success_and_failure`** - Callbacks that execute for both successful and failed requests. These are typically logging/observability tools that need to capture all request data regardless of outcome.

---

## 2. Disable Callbacks

Now that you know which callbacks are active, you can selectively disable them using the `x-litellm-disable-callbacks` header. You can reference any callback name from the list response above.

### Disable a Single Callback

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

### Disable Multiple Callbacks

You can disable multiple callbacks by providing a comma-separated list in the header. Use any combination of callback names from your `/callbacks/list` response.

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

## Header Format and Case Sensitivity

### Expected Header Format

The `x-litellm-disable-callbacks` header accepts callback names in the following formats (use the exact names returned by `/callbacks/list`):

- **Single callback**: `x-litellm-disable-callbacks: langfuse`
- **Multiple callbacks**: `x-litellm-disable-callbacks: langfuse,datadog,prometheus`

When specifying multiple callbacks, use comma-separated values without spaces around the commas.

### Case Sensitivity

**Callback name checks are case insensitive.** This means all of the following are equivalent:

```bash
# These are all equivalent
x-litellm-disable-callbacks: langfuse
x-litellm-disable-callbacks: LANGFUSE  
x-litellm-disable-callbacks: LangFuse
x-litellm-disable-callbacks: langFUSE
```

This applies to both single and multiple callback specifications:

```bash
# Case insensitive for multiple callbacks
x-litellm-disable-callbacks: LANGFUSE,datadog,PROMETHEUS
x-litellm-disable-callbacks: langfuse,DATADOG,prometheus
```


