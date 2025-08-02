import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Computer Use

Computer use allows models to interact with computer interfaces by taking screenshots and performing actions like clicking, typing, and scrolling. This enables AI models to autonomously operate desktop environments.

**Supported Providers:**
- Anthropic API (`anthropic/`)
- Bedrock (Anthropic) (`bedrock/`)
- Vertex AI (Anthropic) (`vertex_ai/`)

**Supported Tool Types:**
- `computer` - Computer interaction tool with display parameters
- `bash` - Bash shell tool  
- `text_editor` - Text editor tool
- `web_search` - Web search tool

LiteLLM will standardize the computer use tools across all supported providers.

## Quick Start

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Computer use tool
    tools = [
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_height_px": 768,
            "display_width_px": 1024,
            "display_number": 0,
        }
    ]
    
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                "text": "Take a screenshot and tell me what you see"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                }
            }
        ]
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy Server">

1. Define computer use models on config.yaml

```yaml
model_list:
  - model_name: claude-3-5-sonnet-latest # Anthropic claude-3-5-sonnet-latest
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: claude-bedrock         # Bedrock Anthropic model
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2
    model_info:
      supports_computer_use: True        # set supports_computer_use to True so /model/info returns this attribute as True
```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Test it using the OpenAI Python SDK

```python
import os 
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", # your litellm proxy api key
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-latest",
    messages=[
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": "Take a screenshot and tell me what you see"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }
    ],
    tools=[
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_height_px": 768,
            "display_width_px": 1024,
            "display_number": 0,
        }
    ]
)

print(response)
```

</TabItem>
</Tabs>

## Checking if a model supports `computer use`

<Tabs>
<TabItem label="LiteLLM Python SDK" value="Python">

Use `litellm.supports_computer_use(model="")` -> returns `True` if model supports computer use and `False` if not

```python
import litellm

assert litellm.supports_computer_use(model="anthropic/claude-3-5-sonnet-latest") == True
assert litellm.supports_computer_use(model="anthropic/claude-3-7-sonnet-20250219") == True
assert litellm.supports_computer_use(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0") == True
assert litellm.supports_computer_use(model="vertex_ai/claude-3-5-sonnet") == True
assert litellm.supports_computer_use(model="openai/gpt-4") == False
```
</TabItem>

<TabItem label="LiteLLM Proxy Server" value="proxy">

1. Define computer use models on config.yaml

```yaml
model_list:
  - model_name: claude-3-5-sonnet-latest # Anthropic claude-3-5-sonnet-latest
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: claude-bedrock         # Bedrock Anthropic model
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2
    model_info:
      supports_computer_use: True        # set supports_computer_use to True so /model/info returns this attribute as True
```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Call `/model_group/info` to check if your model supports `computer use`

```shell
curl -X 'GET' \
  'http://localhost:4000/model_group/info' \
  -H 'accept: application/json' \
  -H 'x-api-key: sk-1234'
```

Expected Response 

```json
{
  "data": [
    {
      "model_group": "claude-3-5-sonnet-latest",
      "providers": ["anthropic"],
      "max_input_tokens": 200000,
      "max_output_tokens": 8192,
      "mode": "chat",
      "supports_computer_use": true, # 👈 supports_computer_use is true
      "supports_vision": true,
      "supports_function_calling": true
    },
    {
      "model_group": "claude-bedrock",
      "providers": ["bedrock"],
      "max_input_tokens": 200000,
      "max_output_tokens": 8192,
      "mode": "chat",
      "supports_computer_use": true, # 👈 supports_computer_use is true
      "supports_vision": true,
      "supports_function_calling": true
    }
  ]
}
```

</TabItem>
</Tabs>

## Different Tool Types

Computer use supports several different tool types for various interaction modes:

<Tabs>
<TabItem value="computer" label="Computer Tool">

The `computer_20241022` tool provides direct screen interaction capabilities.

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "computer_20241022", 
        "name": "computer",
        "display_height_px": 768,
        "display_width_px": 1024,
        "display_number": 0,
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "Click on the search button in the screenshot"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                }
            }
        ]
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

</TabItem>
<TabItem value="bash" label="Bash Tool">

The `bash_20241022` tool provides command line interface access.

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "bash_20241022",
        "name": "bash"
    }
]

messages = [
    {
        "role": "user",
        "content": "List the files in the current directory using bash"
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

</TabItem>
<TabItem value="text_editor" label="Text Editor Tool">

The `text_editor_20250124` tool provides text file editing capabilities.

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "text_editor_20250124",
        "name": "str_replace_editor"
    }
]

messages = [
    {
        "role": "user",
        "content": "Create a simple Python hello world script"
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

</TabItem>
</Tabs>

## Advanced Usage with Multiple Tools

You can combine different computer use tools in a single request:

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "computer_20241022",
        "name": "computer", 
        "display_height_px": 768,
        "display_width_px": 1024,
        "display_number": 0,
    },
    {
        "type": "bash_20241022",
        "name": "bash"
    },
    {
        "type": "text_editor_20250124", 
        "name": "str_replace_editor"
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Take a screenshot, then create a file describing what you see, and finally use bash to show the file contents"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }
    ]
    
response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
            messages=messages,
            tools=tools,
)

print(response)
```

## Spec

### Computer Tool (`computer_20241022`)

```json
{
  "type": "computer_20241022",
  "name": "computer",
  "display_height_px": 768,    // Required: Screen height in pixels  
  "display_width_px": 1024,    // Required: Screen width in pixels
  "display_number": 0          // Optional: Display number (default: 0)
}
```

### Bash Tool (`bash_20241022`)

```json
{
  "type": "bash_20241022", 
  "name": "bash"              // Required: Tool name
}
```

### Text Editor Tool (`text_editor_20250124`)

```json
{
  "type": "text_editor_20250124",
  "name": "str_replace_editor"  // Required: Tool name
}
```

### Web Search Tool (`web_search_20250305`)

```json
{
  "type": "web_search_20250305",
  "name": "web_search"         // Required: Tool name
}
```