import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /generateContent

Use LiteLLM to call Google AI's generateContent endpoints for text generation, multimodal interactions, and streaming responses.

## Overview 

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ |  |
| Logging | ✅ | works across all integrations |
| End-user Tracking | ✅ | |
| Streaming | ✅ | |
| Fallbacks | ✅ | between supported models |
| Loadbalancing | ✅ | between supported models |

## Usage 
---

### LiteLLM Python SDK 

<Tabs>
<TabItem value="basic" label="Basic Usage">

#### Non-streaming example
```python showLineNumbers title="Basic Text Generation"
from litellm.google_genai import agenerate_content
from google.genai.types import ContentDict, PartDict
import os

# Set API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

contents = ContentDict(
    parts=[
        PartDict(text="Hello, can you tell me a short joke?")
    ],
    role="user",
)

response = await agenerate_content(
    contents=contents,
    model="gemini/gemini-2.0-flash",
    max_tokens=100,
)
print(response)
```

#### Streaming example
```python showLineNumbers title="Streaming Text Generation"
from litellm.google_genai import agenerate_content_stream
from google.genai.types import ContentDict, PartDict
import os

# Set API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

contents = ContentDict(
    parts=[
        PartDict(text="Write a long story about space exploration")
    ],
    role="user",
)

response = await agenerate_content_stream(
    contents=contents,
    model="gemini/gemini-2.0-flash",
    max_tokens=500,
)

async for chunk in response:
    print(chunk)
```

</TabItem>

<TabItem value="sync" label="Sync Usage">

#### Sync non-streaming example
```python showLineNumbers title="Sync Text Generation"
from litellm.google_genai import generate_content
from google.genai.types import ContentDict, PartDict
import os

# Set API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

contents = ContentDict(
    parts=[
        PartDict(text="Hello, can you tell me a short joke?")
    ],
    role="user",
)

response = generate_content(
    contents=contents,
    model="gemini/gemini-2.0-flash",
    max_tokens=100,
)
print(response)
```

#### Sync streaming example
```python showLineNumbers title="Sync Streaming Text Generation"
from litellm.google_genai import generate_content_stream
from google.genai.types import ContentDict, PartDict
import os

# Set API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

contents = ContentDict(
    parts=[
        PartDict(text="Write a long story about space exploration")
    ],
    role="user",
)

response = generate_content_stream(
    contents=contents,
    model="gemini/gemini-2.0-flash",
    max_tokens=500,
)

for chunk in response:
    print(chunk)
```

</TabItem>
</Tabs>

### LiteLLM Proxy Server 

1. Setup config.yaml

```yaml
model_list:
    - model_name: gemini-flash
      litellm_params:
        model: gemini/gemini-2.0-flash
        api_key: os.environ/GEMINI_API_KEY
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

<Tabs>
<TabItem value="gemini-proxy" label="Google GenAI SDK">

```python showLineNumbers title="Google GenAI SDK with LiteLLM Proxy"
from google.genai import Client
import os

# Configure Google GenAI SDK to use LiteLLM proxy
os.environ["GOOGLE_GEMINI_BASE_URL"] = "http://localhost:4000"
os.environ["GEMINI_API_KEY"] = "sk-1234"

client = Client()

response = client.models.generate_content(
    model="gemini-flash",
    contents=[
        {
            "parts": [{"text": "Write a short story about AI"}],
            "role": "user"
        }
    ],
    config={"max_output_tokens": 100}
)
```


</TabItem>

<TabItem value="curl-proxy" label="curl">

#### Generate Content

```bash showLineNumbers title="generateContent via LiteLLM Proxy"
curl -L -X POST 'http://localhost:4000/v1beta/models/gemini-flash:generateContent' \
-H 'content-type: application/json' \
-H 'authorization: Bearer sk-1234' \
-d '{
  "contents": [
    {
      "parts": [
        {
          "text": "Write a short story about AI"
        }
      ],
      "role": "user"
    }
  ],
  "generationConfig": {
    "maxOutputTokens": 100
  }
}'
```

#### Stream Generate Content

```bash showLineNumbers title="streamGenerateContent via LiteLLM Proxy"
curl -L -X POST 'http://localhost:4000/v1beta/models/gemini-flash:streamGenerateContent' \
-H 'content-type: application/json' \
-H 'authorization: Bearer sk-1234' \
-d '{
  "contents": [
    {
      "parts": [
        {
          "text": "Write a long story about space exploration"
        }
      ],
      "role": "user"
    }
  ],
  "generationConfig": {
    "maxOutputTokens": 500
  }
}'
```

</TabItem>
</Tabs>


## Related 

- [Use LiteLLM with gemini-cli](../docs/tutorials/litellm_gemini_cli)