# Bedrock - Writer Palmyra

Writer Palmyra X5 and X4 models are available on Amazon Bedrock via the Converse API.

| Model | Context Window | Description |
|-------|---------------|-------------|
| `us.writer.palmyra-x5-v1:0` | 1M tokens | Advanced reasoning, tool calling, document processing |
| `us.writer.palmyra-x4-v1:0` | 128K tokens | Tool calling, document processing |

## Usage

```python
from litellm import completion

response = completion(
    model="bedrock/us.writer.palmyra-x5-v1:0",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

## Tool Calling

```python
from litellm import completion

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = completion(
    model="bedrock/us.writer.palmyra-x5-v1:0",
    messages=[{"role": "user", "content": "What's the weather in Boston?"}],
    tools=tools
)
```

## Document Input

Writer Palmyra models support document inputs (PDF, etc.):

```python
from litellm import completion
import base64

# Read PDF file
with open("document.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode("utf-8")

response = completion(
    model="bedrock/us.writer.palmyra-x5-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:application/pdf;base64,{pdf_data}"
                    }
                },
                {
                    "type": "text",
                    "text": "Summarize this document"
                }
            ]
        }
    ]
)
```

## LiteLLM Proxy Usage

```yaml
model_list:
  - model_name: writer-palmyra-x5
    litellm_params:
      model: bedrock/us.writer.palmyra-x5-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2
```

## Supported Models

| Model ID | Base Model |
|----------|------------|
| `bedrock/us.writer.palmyra-x5-v1:0` | `writer.palmyra-x5-v1:0` |
| `bedrock/us.writer.palmyra-x4-v1:0` | `writer.palmyra-x4-v1:0` |
| `bedrock/writer.palmyra-x5-v1:0` | `writer.palmyra-x5-v1:0` |
| `bedrock/writer.palmyra-x4-v1:0` | `writer.palmyra-x4-v1:0` |

