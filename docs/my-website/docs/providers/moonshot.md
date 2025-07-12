# Moonshot AI
https://platform.moonshot.ai/

**We support ALL Moonshot AI models, just set `moonshot/` as a prefix when sending completion requests**

## API Key
```python
# env variable
os.environ['MOONSHOT_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['MOONSHOT_API_KEY'] = ""
response = completion(
    model="moonshot/moonshot-v1-8k", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['MOONSHOT_API_KEY'] = ""
response = completion(
    model="moonshot/moonshot-v1-8k", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Model Context Windows

| Model Name | Context Window | Function Call |
|------------|----------------|---------------|
| moonshot-v1-8k | 8K tokens | `completion(model="moonshot/moonshot-v1-8k", messages)` |
| moonshot-v1-32k | 32K tokens | `completion(model="moonshot/moonshot-v1-32k", messages)` |
| moonshot-v1-128k | 128K tokens | `completion(model="moonshot/moonshot-v1-128k", messages)` |
| moonshot-v1-auto | 128K tokens | `completion(model="moonshot/moonshot-v1-auto", messages)` |
| kimi-k2 | 128K tokens | `completion(model="moonshot/kimi-k2", messages)` |
| moonshot-v1-32k-0430 | 32K tokens | `completion(model="moonshot/moonshot-v1-32k-0430", messages)` |
| moonshot-v1-128k-0430 | 128K tokens | `completion(model="moonshot/moonshot-v1-128k-0430", messages)` |
| moonshot-v1-8k-0430 | 8K tokens | `completion(model="moonshot/moonshot-v1-8k-0430", messages)` |

## Supported Features
- ✅ Function calling
- ✅ Tool choice (except "required" option)
- ✅ Streaming
- ✅ Temperature control (0-1)
- ❌ Functions parameter (deprecated)
- ❌ Tool choice "required" option

## Important Notes
- Temperature is between 0 and 1
- Temperature close to 0 (<0.3) can only produce n=1 results (automatically handled)
- `tool_choice` doesn't support `required` option (will be dropped or raise error based on `drop_params` setting)
- `functions` parameter is not supported (automatically filtered out - use `tools` instead)
- All models support up to their respective context window limits

## Pricing
For up-to-date pricing information, please refer to the official Moonshot AI pricing page: https://platform.moonshot.ai/docs/pricing