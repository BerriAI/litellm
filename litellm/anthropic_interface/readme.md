## Use LLM API endpoints in Anthropic Interface

Note: This is called `anthropic_interface` because `anthropic` is a known python package and was failing mypy type checking.


## Usage 
---

### LiteLLM Python SDK 

#### Non-streaming example
```python showLineNumbers title="Example using LiteLLM Python SDK"
import litellm
response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    api_key=api_key,
    model="anthropic/claude-3-haiku-20240307",
    max_tokens=100,
)
```

Example response:
```json
{
  "content": [
    {
      "text": "Hi! this is a very short joke",
      "type": "text"
    }
  ],
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "model": "claude-3-7-sonnet-20250219",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503,
    "cache_creation_input_tokens": 2095,
    "cache_read_input_tokens": 0
  }
}
```

#### Streaming example
```python showLineNumbers title="Example using LiteLLM Python SDK"
import litellm
response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    api_key=api_key,
    model="anthropic/claude-3-haiku-20240307",
    max_tokens=100,
    stream=True,
)
async for chunk in response:
    print(chunk)
```

### Optional opening token estimate

When `/v1/messages` streams through the Chat Completions adapter, the opening
`message_start.usage.input_tokens` defaults to `0`. To show an estimated prompt
size while a response streams, enable `anthropic_messages_estimate_initial_input_tokens`.
This setting defaults to `false` and applies to all requests using this adapter

For an existing proxy deployment, add the setting under `litellm_settings` in
`config.yaml` and restart the proxy. Use a YAML boolean, without quotes

```yaml
litellm_settings:
  anthropic_messages_estimate_initial_input_tokens: true
```

For the Python SDK, set the option before making streaming Messages requests

```python
import litellm

litellm.anthropic_messages_estimate_initial_input_tokens = True
```

Send a streaming Messages request to a Chat Completions-backed model, such as a
`hosted_vllm/` model. The opening `message_start` event includes the estimated
input count. If estimation fails, the count remains `0`. Set the option to
`false` in YAML or `False` in Python to disable estimation and skip token counting

The opening count is approximate and is not suitable for billing or enforcing
exact context limits. Opening output and cache counts remain `0`. Final
`message_delta` usage still comes from the provider and takes precedence over the
opening estimate. Native Anthropic streams, the Responses adapter, and
non-streaming responses are unaffected. This setting does not select the adapter

### LiteLLM Proxy Server 


1. Setup config.yaml

```yaml
model_list:
    - model_name: anthropic-claude
      litellm_params:
        model: claude-3-7-sonnet-latest
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

<Tabs>
<TabItem label="Anthropic Python SDK" value="python">

```python showLineNumbers title="Example using LiteLLM Proxy Server"
import anthropic

# point anthropic sdk to litellm proxy 
client = anthropic.Anthropic(
    base_url="http://0.0.0.0:4000",
    api_key="<your-master-key>",
)

response = client.messages.create(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="anthropic/claude-3-haiku-20240307",
    max_tokens=100,
)
```
</TabItem>
<TabItem label="curl" value="curl">

```bash showLineNumbers title="Example using LiteLLM Proxy Server"
curl -L -X POST 'http://0.0.0.0:4000/v1/messages' \
-H 'content-type: application/json' \
-H 'x-api-key: $LITELLM_API_KEY' \
-H 'anthropic-version: 2023-06-01' \
-d '{
  "model": "anthropic-claude",
  "messages": [
    {
      "role": "user",
      "content": "Hello, can you tell me a short joke?"
    }
  ],
  "max_tokens": 100
}'
```
