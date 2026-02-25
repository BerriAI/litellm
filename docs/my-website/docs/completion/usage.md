# Usage 

LiteLLM returns the OpenAI compatible usage object across all providers.

```bash
"usage": {
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int
  }
```

## Quick Start 

```python
from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

response = completion(
  model="gpt-3.5-turbo",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)

print(response.usage)
```
> **Note:** LiteLLM supports endpoint bridging—if a model does not natively support a requested endpoint, LiteLLM will automatically route the call to the correct supported endpoint (such as bridging `/chat/completions` to `/responses` or vice versa) based on the model's `mode`set in `model_prices_and_context_window`.

## Streaming Usage

if `stream_options={"include_usage": True}` is set, an additional chunk will be streamed before the data: [DONE] message. The usage field on this chunk shows the token usage statistics for the entire request, and the choices field will always be an empty array. All other chunks will also include a usage field, but with a null value.


```python
from litellm import completion 

completion = completion(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  stream=True,
  stream_options={"include_usage": True}
)

for chunk in completion:
  print(chunk.choices[0].delta)

```

### Proxy: Always Include Streaming Usage

When using the LiteLLM Proxy, you can configure it to automatically include usage information in all streaming responses, even if the client doesn't send `stream_options={"include_usage": True}`.

#### Configuration

Add the following to your config.yaml:

```yaml
general_settings:
  always_include_stream_usage: true
```

Alternatively, configure it through the UI:

1. Navigate to the LiteLLM Proxy UI
2. Go to `Settings` >  `Router Settings` > `General`
3. Find the `always_include_stream_usage` setting
4. Toggle it to `true`
5. Click `Update` to save

#### How it works

When `always_include_stream_usage` is enabled:
- All streaming requests will automatically have `stream_options={"include_usage": True}` added
- Clients will receive usage information in the final chunk, even if they didn't explicitly request it
- If a client already provides `stream_options`, `include_usage: True` will be added without overwriting other options
- Non-streaming requests are not affected

#### Example

With this setting enabled, a simple streaming request like:

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Will automatically receive usage information in the response, without needing to explicitly include `stream_options`.

```
