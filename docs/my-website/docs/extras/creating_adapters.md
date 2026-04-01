# Call any LiteLLM model in your custom format

Use this to call any LiteLLM supported `.completion()` model, in your custom format. Useful if you have a custom API and want to support any LiteLLM supported model.

## How it works

Your request → Adapter translates to OpenAI format → LiteLLM processes it → Adapter translates response back → Your response

## Create an Adapter

Inherit from `CustomLogger` and implement 3 methods:

```python
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import ChatCompletionRequest
from litellm.types.utils import ModelResponse

class MyAdapter(CustomLogger):
    def translate_completion_input_params(self, kwargs) -> ChatCompletionRequest:
        """Convert your format → OpenAI format"""
        # Example: Anthropic to OpenAI
        return {
            "model": kwargs["model"],
            "messages": self._convert_messages(kwargs["messages"]),
            "max_tokens": kwargs.get("max_tokens"),
        }

    def translate_completion_output_params(self, response: ModelResponse):
        """Convert OpenAI format → your format"""
        # Return your provider's response format
        return MyProviderResponse(
            id=response.id,
            content=response.choices[0].message.content,
            usage=response.usage,
        )

    def translate_completion_output_params_streaming(self, completion_stream):
        """Handle streaming responses"""
        return MyStreamWrapper(completion_stream)
```

## Register it

```python
import litellm

my_adapter = MyAdapter()
litellm.adapters = [{"id": "my_provider", "adapter": my_adapter}]
```

## Use it

```python
from litellm import adapter_completion

# Now you can use your provider's format with any LiteLLM model
response = adapter_completion(
    adapter_id="my_provider",
    model="gpt-4",  # or any LiteLLM model
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=100
)
```

### Streaming

```python
stream = adapter_completion(
    adapter_id="my_provider",
    model="gpt-4",
    messages=[{"role": "user", "content": "hello"}],
    stream=True
)

for chunk in stream:
    print(chunk)
```

### Async

```python
from litellm import aadapter_completion

response = await aadapter_completion(
    adapter_id="my_provider",
    model="gpt-4",
    messages=[{"role": "user", "content": "hello"}]
)
```

## Example: Anthropic Adapter

Here's how we translate Anthropic's format:

### Input Translation

```python
def translate_completion_input_params(self, kwargs):
    model = kwargs.pop("model")
    messages = kwargs.pop("messages")
    
    # Convert Anthropic messages to OpenAI format
    openai_messages = []
    for msg in messages:
        if msg["role"] == "user":
            openai_messages.append({
                "role": "user",
                "content": msg["content"]
            })
    
    # Handle system message
    if "system" in kwargs:
        openai_messages.insert(0, {
            "role": "system",
            "content": kwargs.pop("system")
        })
    
    return {
        "model": model,
        "messages": openai_messages,
        **kwargs  # pass through other params
    }
```

### Output Translation

```python
def translate_completion_output_params(self, response):
    return AnthropicResponse(
        id=response.id,
        type="message",
        role="assistant",
        content=[{
            "type": "text",
            "text": response.choices[0].message.content
        }],
        usage={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
    )
```

### Streaming

```python
from litellm.types.utils import AdapterCompletionStreamWrapper

class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    def __init__(self, completion_stream, model):
        super().__init__(completion_stream)
        self.model = model
        self.first_chunk = True
    
    async def __anext__(self):
        # First chunk
        if self.first_chunk:
            self.first_chunk = False
            return {"type": "message_start", "message": {...}}
        
        # Stream chunks
        async for chunk in self.completion_stream:
            return {
                "type": "content_block_delta",
                "delta": {"text": chunk.choices[0].delta.content}
            }
        
        # Last chunk
        return {"type": "message_stop"}

def translate_completion_output_params_streaming(self, stream, model):
    return AnthropicStreamWrapper(stream, model)
```

## Use with Proxy

Add to your proxy config:

```yaml
general_settings:
  pass_through_endpoints:
    - path: "/v1/messages"
      target: "my_module.MyAdapter"
```

Then call it:

```bash
curl http://localhost:4000/v1/messages \
  -H "Authorization: Bearer sk-1234" \
  -d '{"model": "gpt-4", "messages": [...]}'
```

## Real Example

Check out the full Anthropic adapter:
- [transformation.py](https://github.com/BerriAI/litellm/blob/main/litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py)
- [handler.py](https://github.com/BerriAI/litellm/blob/main/litellm/llms/anthropic/experimental_pass_through/adapters/handler.py)
- [streaming_iterator.py](https://github.com/BerriAI/litellm/blob/main/litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py)

## That's it

1. Create a class that inherits `CustomLogger`
2. Implement the 3 translation methods
3. Register with `litellm.adapters = [{"id": "...", "adapter": ...}]`
4. Call with `adapter_completion(adapter_id="...")`
