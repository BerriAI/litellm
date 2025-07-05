import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Google AI generateContent

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
| Multimodal | ✅ | images, audio, video, PDF |
| Function Calling | ✅ | |
| JSON Mode | ✅ | |
| System Instructions | ✅ | |

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

<TabItem value="vertex" label="Vertex AI">

#### Vertex AI example
```python showLineNumbers title="Vertex AI Usage"
from litellm.google_genai import agenerate_content
from google.genai.types import ContentDict, PartDict
import os

# Set Vertex AI credentials
os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

contents = ContentDict(
    parts=[
        PartDict(text="Hello, can you tell me a short joke?")
    ],
    role="user",
)

response = await agenerate_content(
    contents=contents,
    model="vertex_ai/gemini-2.0-flash",
    max_tokens=100,
)
print(response)
```

</TabItem>
</Tabs>

### LiteLLM Proxy Server 

<Tabs>
<TabItem value="gemini-proxy" label="Google AI Studio">

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

```python showLineNumbers title="Google AI Studio Example using LiteLLM Proxy Server"
import openai

# point openai sdk to litellm proxy 
client = openai.OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a short story about AI"}],
    model="gemini-flash",
    max_tokens=100,
)
```

</TabItem>

<TabItem value="vertex-proxy" label="Vertex AI">

1. Setup config.yaml

```yaml
model_list:
    - model_name: vertex-gemini
      litellm_params:
        model: vertex_ai/gemini-2.0-flash
        vertex_project: your-gcp-project-id
        vertex_location: us-central1
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python showLineNumbers title="Vertex AI Example using LiteLLM Proxy Server"
import openai

# point openai sdk to litellm proxy 
client = openai.OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a short story about AI"}],
    model="vertex-gemini",
    max_tokens=100,
)
```

</TabItem>

<TabItem value="curl-proxy" label="curl">

```bash showLineNumbers title="Example using LiteLLM Proxy Server"
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'content-type: application/json' \
-H 'authorization: Bearer sk-1234' \
-d '{
  "model": "gemini-flash",
  "messages": [
    {
      "role": "user",
      "content": "Write a short story about AI"
    }
  ],
  "max_tokens": 100
}'
```

</TabItem>
</Tabs>


## Request Format
---

### Google GenAI SDK Format

For the direct Google GenAI SDK, use `ContentDict` and `PartDict`:

```python
from google.genai.types import ContentDict, PartDict

contents = ContentDict(
    parts=[
        PartDict(text="Hello, world")
    ],
    role="user",
)
```


## Response Format
---

### Google GenAI SDK Response

Direct SDK returns Google's native response format:

```python
# GenerateContentResponse object
print(response.model_dump_json(indent=4))
```


## Related 

- [Use LiteLLM with gemini-cli](../docs/tutorials/litellm_gemini_cli)