import Image from '@theme/IdealImage';

# Prompt Management

LiteLLM supports using [Langfuse](https://langfuse.com/docs/prompts/get-started) for prompt management on the proxy.

## Quick Start

1. Add Langfuse as a 'callback' in your config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/chatgpt-v-2
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE

litellm_settings:
    callbacks: ["langfuse"] # 👈 KEY CHANGE
```

2. Start the proxy

```bash
litellm-proxy --config config.yaml
```

3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "THIS WILL BE IGNORED"
        }
    ],
    "metadata": {
        "langfuse_prompt_id": "value",
        "langfuse_prompt_variables": { # [OPTIONAL]
            "key": "value"
        }
    }
}'
```

## What is 'langfuse_prompt_id'?

- `langfuse_prompt_id`: The ID of the prompt that will be used for the request.

<Image img={require('../../img/langfuse_prompt_id.png')} />

## What will the formatted prompt look like?

### `/chat/completions` messages

The message will be added to the start of the prompt.

- if the Langfuse prompt is a list, it will be added to the start of the messages list (assuming it's an OpenAI compatible message).

- if the Langfuse prompt is a string, it will be added as a system message.

```python
if isinstance(compiled_prompt, list):
    data["messages"] = compiled_prompt + data["messages"]
else:
    data["messages"] = [
        {"role": "system", "content": compiled_prompt}
    ] + data["messages"]
```

### `/completions` messages

The message will be added to the start of the prompt.

```python
data["prompt"] = compiled_prompt + "\n" + data["prompt"]
```