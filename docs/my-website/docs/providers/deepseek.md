import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Deepseek
https://deepseek.com/

**We support ALL Deepseek models, just set `deepseek/` as a prefix when sending completion requests**

## API Key
```python
# env variable
os.environ['DEEPSEEK_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['DEEPSEEK_API_KEY'] = ""
response = completion(
    model="deepseek/deepseek-chat", 
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

os.environ['DEEPSEEK_API_KEY'] = ""
response = completion(
    model="deepseek/deepseek-chat", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models - ALL Deepseek Models Supported!
We support ALL Deepseek models, just set `deepseek/` as a prefix when sending completion requests

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| deepseek-chat | `completion(model="deepseek/deepseek-chat", messages)` | 
| deepseek-coder | `completion(model="deepseek/deepseek-coder", messages)` | 


## Reasoning Models
| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| deepseek-reasoner | `completion(model="deepseek/deepseek-reasoner", messages)` | 

## Tool calling with thinking / `reasoning_content`

When you enable DeepSeek thinking mode (`thinking: {"type": "enabled"}`), the API requires you to echo back `reasoning_content` on assistant messages that contain tool calls. If you omit it, DeepSeek returns a 400 error. See the official docs: https://api-docs.deepseek.com/guides/thinking_mode.

```python
from litellm import completion
import json
import os

os.environ["DEEPSEEK_API_KEY"] = ""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

# First turn: model decides whether to call tools; include thinking
resp = completion(
    model="deepseek/deepseek-reasoner",
    messages=messages,
    tools=tools,
    thinking={"type": "enabled"}
)
assistant_msg = resp.choices[0].message

# Echo reasoning_content back with the tool call (required by DeepSeek)
messages.append(
    {
        "role": "assistant",
        "content": assistant_msg.content or "",
        "tool_calls": assistant_msg.tool_calls,
        "reasoning_content": assistant_msg.reasoning_content,
    }
)

# Execute the tool(s)
for tool_call in assistant_msg.tool_calls or []:
    args = json.loads(tool_call.function.arguments or "{}")
    tool_result = f"Mock weather for {args.get('city', 'unknown')}"
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        }
    )

# Second turn: send tool results + reasoning_content back to get the final answer
resp = completion(
    model="deepseek/deepseek-reasoner",
    messages=messages,
  thinking={"type": "enabled"}
)
print(resp.choices[0].message.content)

# If you start a new user question, drop old reasoning_content to save bandwidth:
# for m in messages:
#     if isinstance(m, dict) and m.get("role") == "assistant":
#         m.pop("reasoning_content", None)
```



<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ['DEEPSEEK_API_KEY'] = ""
resp = completion(
    model="deepseek/deepseek-reasoner",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)

print(
    resp.choices[0].message.reasoning_content
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: deepseek-reasoner
    litellm_params:
        model: deepseek/deepseek-reasoner
        api_key: os.environ/DEEPSEEK_API_KEY
```

2. Run proxy

```bash
python litellm/proxy/main.py
```

3. Test it!

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "deepseek-reasoner",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hi, how are you ?"
          }
        ]
      }
    ]
}'
```

</TabItem>

</Tabs>
