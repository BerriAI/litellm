import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 'Thinking' / 'Reasoning Content'

:::info

Requires LiteLLM v1.63.0+

:::

Supported Providers:
- Deepseek (`deepseek/`)
- Anthropic API (`anthropic/`)
- Bedrock (Anthropic + Deepseek + GPT-OSS) (`bedrock/`)
- Vertex AI (Anthropic) (`vertexai/`)
- OpenRouter (`openrouter/`)
- XAI (`xai/`)
- Google AI Studio (`google/`)
- Vertex AI (`vertex_ai/`)
- Perplexity (`perplexity/`)
- Mistral AI (Magistral models) (`mistral/`)
- Groq (`groq/`)

LiteLLM will standardize the `reasoning_content` in the response and `thinking_blocks` in the assistant message.

```python title="Example response from litellm"
"message": {
    ...
    "reasoning_content": "The capital of France is Paris.",
    "thinking_blocks": [ # only returned for Anthropic models
        {
            "type": "thinking",
            "thinking": "The capital of France is Paris.",
            "signature": "EqoBCkgIARABGAIiQL2UoU0b1OHYi+..."
        }
    ]
}
```

## Quick Start 

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import completion
import os 

os.environ["ANTHROPIC_API_KEY"] = ""

response = completion(
  model="anthropic/claude-3-7-sonnet-20250219",
  messages=[
    {"role": "user", "content": "What is the capital of France?"},
  ],
  reasoning_effort="low", 
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "anthropic/claude-3-7-sonnet-20250219",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "reasoning_effort": "low"
}'
```
</TabItem>
</Tabs>

**Expected Response**

```bash
{
    "id": "3b66124d79a708e10c603496b363574c",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": " won the FIFA World Cup in 2022.",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "created": 1723323084,
    "model": "deepseek/deepseek-chat",
    "object": "chat.completion",
    "system_fingerprint": "fp_7e0991cad4",
    "usage": {
        "completion_tokens": 12,
        "prompt_tokens": 16,
        "total_tokens": 28,
    },
    "service_tier": null
}
```

## Tool Calling with `thinking`

Here's how to use `thinking` blocks by Anthropic with tool calling.

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
litellm._turn_on_debug()
litellm.modify_params = True
model = "anthropic/claude-3-7-sonnet-20250219" # works across Anthropic, Bedrock, Vertex AI
# Step 1: send the conversation and available functions to the model
messages = [
    {
        "role": "user",
        "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
    }
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]
response = litellm.completion(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
    reasoning_effort="low",
)
print("Response\n", response)
response_message = response.choices[0].message
tool_calls = response_message.tool_calls

print("Expecting there to be 3 tool calls")
assert (
    len(tool_calls) > 0
)  # this has to call the function for SF, Tokyo and paris

# Step 2: check if the model wanted to call a function
print(f"tool_calls: {tool_calls}")
if tool_calls:
    # Step 3: call the function
    # Note: the JSON response may not always be valid; be sure to handle errors
    available_functions = {
        "get_current_weather": get_current_weather,
    }  # only one function in this example, but you can have multiple
    messages.append(
        response_message
    )  # extend conversation with assistant's reply
    print("Response message\n", response_message)
    # Step 4: send the info for each function call and function response to the model
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        if function_name not in available_functions:
            # the model called a function that does not exist in available_functions - don't try calling anything
            return
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
    print(f"messages: {messages}")
    second_response = litellm.completion(
        model=model,
        messages=messages,
        seed=22,
        reasoning_effort="low",
        # tools=tools,
        drop_params=True,
    )  # get a new response from the model where it can see the function response
    print("second response\n", second_response)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml showLineNumbers
model_list:
  - model_name: claude-3-7-sonnet-thinking
    litellm_params:
      model: anthropic/claude-3-7-sonnet-20250219
      api_key: os.environ/ANTHROPIC_API_KEY
      thinking: {
        "type": "enabled",
        "budget_tokens": 1024
      }
```

2. Run proxy

```bash showLineNumbers
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Make 1st call

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "claude-3-7-sonnet-thinking",
    "messages": [
      {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses"},
    ],
    "tools": [
        {
          "type": "function",
          "function": {
              "name": "get_current_weather",
              "description": "Get the current weather in a given location",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "location": {
                          "type": "string",
                          "description": "The city and state",
                      },
                      "unit": {
                          "type": "string",
                          "enum": ["celsius", "fahrenheit"],
                      },
                  },
                  "required": ["location"],
              },
          },
        }
    ],
    "tool_choice": "auto"
  }'
```

4. Make 2nd call with tool call results

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "claude-3-7-sonnet-thinking",
    "messages": [
      {
        "role": "user",
        "content": "What\'s the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses"
      },
      {
        "role": "assistant",
        "content": "I\'ll check the current weather for these three cities for you:",
        "tool_calls": [
          {
            "index": 2,
            "function": {
              "arguments": "{\"location\": \"San Francisco\"}",
              "name": "get_current_weather"
            },
            "id": "tooluse_mnqzmtWYRjCxUInuAdK7-w",
            "type": "function"
          }
        ],
        "function_call": null,
        "reasoning_content": "The user is asking for the current weather in three different locations: San Francisco, Tokyo, and Paris. I have access to the `get_current_weather` function that can provide this information.\n\nThe function requires a `location` parameter, and has an optional `unit` parameter. The user hasn't specified which unit they prefer (celsius or fahrenheit), so I'll use the default provided by the function.\n\nI need to make three separate function calls, one for each location:\n1. San Francisco\n2. Tokyo\n3. Paris\n\nThen I'll compile the results into a response with three distinct weather reports as requested by the user.",
        "thinking_blocks": [
          {
            "type": "thinking",
            "thinking": "The user is asking for the current weather in three different locations: San Francisco, Tokyo, and Paris. I have access to the `get_current_weather` function that can provide this information.\n\nThe function requires a `location` parameter, and has an optional `unit` parameter. The user hasn't specified which unit they prefer (celsius or fahrenheit), so I'll use the default provided by the function.\n\nI need to make three separate function calls, one for each location:\n1. San Francisco\n2. Tokyo\n3. Paris\n\nThen I'll compile the results into a response with three distinct weather reports as requested by the user.",
            "signature": "EqoBCkgIARABGAIiQCkBXENoyB+HstUOs/iGjG+bvDbIQRrxPsPpOSt5yDxX6iulZ/4K/w9Rt4J5Nb2+3XUYsyOH+CpZMfADYvItFR4SDPb7CmzoGKoolCMAJRoM62p1ZRASZhrD3swqIjAVY7vOAFWKZyPEJglfX/60+bJphN9W1wXR6rWrqn3MwUbQ5Mb/pnpeb10HMploRgUqEGKOd6fRKTkUoNDuAnPb55c="
          }
        ],
        "provider_specific_fields": {
          "reasoningContentBlocks": [
            {
              "reasoningText": {
                "signature": "EqoBCkgIARABGAIiQCkBXENoyB+HstUOs/iGjG+bvDbIQRrxPsPpOSt5yDxX6iulZ/4K/w9Rt4J5Nb2+3XUYsyOH+CpZMfADYvItFR4SDPb7CmzoGKoolCMAJRoM62p1ZRASZhrD3swqIjAVY7vOAFWKZyPEJglfX/60+bJphN9W1wXR6rWrqn3MwUbQ5Mb/pnpeb10HMploRgUqEGKOd6fRKTkUoNDuAnPb55c=",
                "text": "The user is asking for the current weather in three different locations: San Francisco, Tokyo, and Paris. I have access to the `get_current_weather` function that can provide this information.\n\nThe function requires a `location` parameter, and has an optional `unit` parameter. The user hasn't specified which unit they prefer (celsius or fahrenheit), so I'll use the default provided by the function.\n\nI need to make three separate function calls, one for each location:\n1. San Francisco\n2. Tokyo\n3. Paris\n\nThen I'll compile the results into a response with three distinct weather reports as requested by the user."
              }
            }
          ]
        }
      },
      {
        "tool_call_id": "tooluse_mnqzmtWYRjCxUInuAdK7-w",
        "role": "tool",
        "name": "get_current_weather",
        "content": "{\"location\": \"San Francisco\", \"temperature\": \"72\", \"unit\": \"fahrenheit\"}"
      }
    ]
  }'
```

</TabItem>
</Tabs>

## Switching between Anthropic + Deepseek models 

Set `drop_params=True` to drop the 'thinking' blocks when swapping from Anthropic to Deepseek models. Suggest improvements to this approach [here](https://github.com/BerriAI/litellm/discussions/8927).

```python showLineNumbers
litellm.drop_params = True # 👈 EITHER GLOBALLY or per request

# or per request
## Anthropic
response = litellm.completion(
  model="anthropic/claude-3-7-sonnet-20250219",
  messages=[{"role": "user", "content": "What is the capital of France?"}],
  reasoning_effort="low",
  drop_params=True,
)

## Deepseek
response = litellm.completion(
  model="deepseek/deepseek-chat",
  messages=[{"role": "user", "content": "What is the capital of France?"}],
  reasoning_effort="low",
  drop_params=True,
)
```

## Spec 


These fields can be accessed via `response.choices[0].message.reasoning_content` and `response.choices[0].message.thinking_blocks`.

- `reasoning_content` - str: The reasoning content from the model. Returned across all providers.
- `thinking_blocks` - Optional[List[Dict[str, str]]]: A list of thinking blocks from the model. Only returned for Anthropic models.
  - `type` - str: The type of thinking block.
  - `thinking` - str: The thinking from the model.
  - `signature` - str: The signature delta from the model.



## Pass `thinking` to Anthropic models

You can also pass the `thinking` parameter to Anthropic models.

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
response = litellm.completion(
  model="anthropic/claude-3-7-sonnet-20250219",
  messages=[{"role": "user", "content": "What is the capital of France?"}],
  thinking={"type": "enabled", "budget_tokens": 1024},
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "anthropic/claude-3-7-sonnet-20250219",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "thinking": {"type": "enabled", "budget_tokens": 1024}
  }'
```

</TabItem>
</Tabs>

## Checking if a model supports reasoning

<Tabs>
<TabItem label="LiteLLM Python SDK" value="Python">

Use `litellm.supports_reasoning(model="")` -> returns `True` if model supports reasoning and `False` if not.

```python showLineNumbers title="litellm.supports_reasoning() usage"
import litellm 

# Example models that support reasoning
assert litellm.supports_reasoning(model="anthropic/claude-3-7-sonnet-20250219") == True
assert litellm.supports_reasoning(model="deepseek/deepseek-chat") == True 

# Example models that do not support reasoning
assert litellm.supports_reasoning(model="openai/gpt-3.5-turbo") == False 
```
</TabItem>

<TabItem label="LiteLLM Proxy Server" value="proxy">

1. Define models that support reasoning in your `config.yaml`. You can optionally add `supports_reasoning: True` to the `model_info` if LiteLLM does not automatically detect it for your custom model.

```yaml showLineNumbers title="litellm proxy config.yaml"
model_list:
  - model_name: claude-3-sonnet-reasoning
    litellm_params:
      model: anthropic/claude-3-7-sonnet-20250219
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: deepseek-reasoning
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY
  # Example for a custom model where detection might be needed
  - model_name: my-custom-reasoning-model 
    litellm_params:
      model: openai/my-custom-model # Assuming it's OpenAI compatible
      api_base: http://localhost:8000
      api_key: fake-key
    model_info:
      supports_reasoning: True # Explicitly mark as supporting reasoning
```

2. Run the proxy server:

```bash showLineNumbers title="litellm --config config.yaml"
litellm --config config.yaml
```

3. Call `/model_group/info` to check if your model supports `reasoning`

```shell showLineNumbers title="curl /model_group/info"
curl -X 'GET' \
  'http://localhost:4000/model_group/info' \
  -H 'accept: application/json' \
  -H 'x-api-key: sk-1234'
```

Expected Response 

```json showLineNumbers title="response from /model_group/info"
{
  "data": [
    {
      "model_group": "claude-3-sonnet-reasoning",
      "providers": ["anthropic"],
      "mode": "chat",
      "supports_reasoning": true,
    },
    {
      "model_group": "deepseek-reasoning",
      "providers": ["deepseek"],
      "supports_reasoning": true,
    },
    {
      "model_group": "my-custom-reasoning-model",
      "providers": ["openai"],
      "supports_reasoning": true,
    }
  ]
}
````


</TabItem>
</Tabs>
