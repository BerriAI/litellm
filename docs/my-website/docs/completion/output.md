# Output 

## Format
Here's the exact json output and type you can expect from all litellm `completion` calls for all models

```python 
{
  'choices': [
    {
      'finish_reason': str,     # String: 'stop'
      'index': int,             # Integer: 0
      'message': {              # Dictionary [str, str]
        'role': str,            # String: 'assistant'
        'content': str          # String: "default message"
      }
    }
  ],
  'created': str,               # String: None
  'model': str,                 # String: None
  'usage': {                    # Dictionary [str, int]
    'prompt_tokens': int,       # Integer
    'completion_tokens': int,   # Integer
    'total_tokens': int         # Integer
  }
}

```

You can access the response as a dictionary or as a class object, just as OpenAI allows you
```python
print(response.choices[0].message.content)
print(response['choices'][0]['message']['content'])
```

Here's what an example response looks like 
```python
{
  'choices': [
     {
        'finish_reason': 'stop',
        'index': 0,
        'message': {
           'role': 'assistant',
            'content': " I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
        }
      }
    ],
 'created': 1691429984.3852863,
 'model': 'claude-instant-1',
 'usage': {'prompt_tokens': 18, 'completion_tokens': 23, 'total_tokens': 41}
}
```

## Native Finish Reason

LiteLLM maps all provider-specific `finish_reason` values to OpenAI-compatible values (`stop`, `length`, `tool_calls`, `function_call`, `content_filter`). When the original provider value differs from the mapped value, it is preserved in `provider_specific_fields["native_finish_reason"]`.

This is useful for agent loops that need to distinguish between different stop conditions (e.g., Gemini's `MALFORMED_FUNCTION_CALL` vs a normal `stop`).

```python
response = completion(model="gemini/gemini-2.0-flash", messages=messages)

choice = response.choices[0]
print(choice.finish_reason)  # "stop" (OpenAI-compatible)

# Access the original provider value when it differs:
if hasattr(choice, "provider_specific_fields") and choice.provider_specific_fields:
    native = choice.provider_specific_fields.get("native_finish_reason")
    if native == "MALFORMED_FUNCTION_CALL":
        # Handle malformed function call differently from a normal stop
        pass
```

When the provider already returns an OpenAI-compatible value (e.g., `stop`), `native_finish_reason` is not set.

## Additional Attributes

You can also access information like latency. 

```python
from litellm import completion
import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages=[{"role": "user", "content": "Hey!"}]

response = completion(model="claude-2", messages=messages)

print(response.response_ms) # 616.25# 616.25
```