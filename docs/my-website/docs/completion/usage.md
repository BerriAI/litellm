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
