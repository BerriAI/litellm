# Anthropic
LiteLLM supports

- `claude-2`
- `claude-2.1`
- `claude-instant-1`
- `claude-instant-1.2`

## API Keys

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
```

## Usage

```python
import os
from litellm import completion

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-instant-1", messages=messages)
print(response)
```

## Usage - "Assistant Pre-fill"

You can "put words in Claude's mouth" by including an `assistant` role message as the last item in the `messages` array.

> [!IMPORTANT]
> The returned completion will _not_ include your "pre-fill" text, since it is part of the prompt itself. Make sure to prefix Claude's completion with your pre-fill.

```python
import os
from litellm import completion

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [
    {"role": "user", "content": "How do you say 'Hello' in German? Return your answer as a JSON object, like this:\n\n{ \"Hello\": \"Hallo\" }"},
    {"role": "assistant", "content": "{"},
]
response = completion(model="claude-2.1", messages=messages)
print(response)
```

### Example prompt sent to Claude

```

Human: How do you say 'Hello' in German? Return your answer as a JSON object, like this:

{ "Hello": "Hallo" }

Assistant: {
```

## Usage - "System" messages
If you're using Anthropic's Claude 2.1 with Bedrock, `system` role messages are properly formatted for you.

```python
import os
from litellm import completion

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [
    {"role": "system", "content": "You are a snarky assistant."},
    {"role": "user", "content": "How do I boil water?"},
]
response = completion(model="claude-2.1", messages=messages)
```

### Example prompt sent to Claude

```
You are a snarky assistant.

Human: How do I boil water?

Assistant:
```

## Streaming
Just set `stream=True` when calling completion.

```python
import os
from litellm import completion

# set env
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-instant-1", messages=messages, stream=True)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```


### Model Details

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| claude-2.1  | `completion('claude-2.1', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-2  | `completion('claude-2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-instant-1  | `completion('claude-instant-1', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-instant-1.2  | `completion('claude-instant-1.2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
