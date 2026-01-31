import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /v1/messages

Use LiteLLM to call all your LLM APIs in the Anthropic `v1/messages` format. 


## Overview 

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ |  |
| Logging | ✅ | works across all integrations |
| End-user Tracking | ✅ | |
| Streaming | ✅ | |
| Fallbacks | ✅ | between supported models |
| Loadbalancing | ✅ | between supported models |
| Guardrails | ✅ | |
| Support llm providers | **All LiteLLM supported providers** | `openai`, `anthropic`, `bedrock`, `vertex_ai`, `gemini`, `azure`, `azure_ai`, etc. |

## Usage 
---

### LiteLLM Python SDK 

<Tabs>
<TabItem value="anthropic" label="Anthropic">

#### Non-streaming example
```python showLineNumbers title="Anthropic Example using LiteLLM Python SDK"
import litellm
response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    api_key=api_key,
    model="anthropic/claude-3-haiku-20240307",
    max_tokens=100,
)
```

#### Streaming example
```python showLineNumbers title="Anthropic Streaming Example using LiteLLM Python SDK"
import litellm
response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    api_key=api_key,
    model="anthropic/claude-3-haiku-20240307",
    max_tokens=100,
    stream=True,
)
async for chunk in response:
    print(chunk)
```

</TabItem>

<TabItem value="openai" label="OpenAI">

#### Non-streaming example
```python showLineNumbers title="OpenAI Example using LiteLLM Python SDK"
import litellm
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="openai/gpt-4",
    max_tokens=100,
)
```

#### Streaming example
```python showLineNumbers title="OpenAI Streaming Example using LiteLLM Python SDK"
import litellm
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="openai/gpt-4",
    max_tokens=100,
    stream=True,
)
async for chunk in response:
    print(chunk)
```

</TabItem>

<TabItem value="gemini" label="Google AI Studio">

#### Non-streaming example
```python showLineNumbers title="Google Gemini Example using LiteLLM Python SDK"
import litellm
import os

# Set API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="gemini/gemini-2.0-flash-exp",
    max_tokens=100,
)
```

#### Streaming example
```python showLineNumbers title="Google Gemini Streaming Example using LiteLLM Python SDK"
import litellm
import os

# Set API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="gemini/gemini-2.0-flash-exp",
    max_tokens=100,
    stream=True,
)
async for chunk in response:
    print(chunk)
```

</TabItem>

<TabItem value="vertex" label="Vertex AI">

#### Non-streaming example
```python showLineNumbers title="Vertex AI Example using LiteLLM Python SDK"
import litellm
import os

# Set credentials - Vertex AI uses application default credentials
# Run 'gcloud auth application-default login' to authenticate
os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="vertex_ai/gemini-2.0-flash-exp",
    max_tokens=100,
)
```

#### Streaming example
```python showLineNumbers title="Vertex AI Streaming Example using LiteLLM Python SDK"
import litellm
import os

# Set credentials - Vertex AI uses application default credentials
# Run 'gcloud auth application-default login' to authenticate
os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="vertex_ai/gemini-2.0-flash-exp",
    max_tokens=100,
    stream=True,
)
async for chunk in response:
    print(chunk)
```

</TabItem>

<TabItem value="bedrock" label="AWS Bedrock">

#### Non-streaming example
```python showLineNumbers title="AWS Bedrock Example using LiteLLM Python SDK"
import litellm
import os

# Set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key-id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-access-key"
os.environ["AWS_REGION_NAME"] = "us-west-2"  # or your AWS region

response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=100,
)
```

#### Streaming example
```python showLineNumbers title="AWS Bedrock Streaming Example using LiteLLM Python SDK"
import litellm
import os

# Set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key-id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-access-key"
os.environ["AWS_REGION_NAME"] = "us-west-2"  # or your AWS region

response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=100,
    stream=True,
)
async for chunk in response:
    print(chunk)
```

</TabItem>
</Tabs>

Example response:
```json
{
  "content": [
    {
      "text": "Hi! this is a very short joke",
      "type": "text"
    }
  ],
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "model": "claude-3-7-sonnet-20250219",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503,
    "cache_creation_input_tokens": 2095,
    "cache_read_input_tokens": 0
  }
}
```

### LiteLLM Proxy Server 

<Tabs>
<TabItem value="anthropic-proxy" label="Anthropic">

1. Setup config.yaml

```yaml
model_list:
    - model_name: anthropic-claude
      litellm_params:
        model: claude-3-7-sonnet-latest
        api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python showLineNumbers title="Anthropic Example using LiteLLM Proxy Server"
import anthropic

# point anthropic sdk to litellm proxy 
client = anthropic.Anthropic(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

response = client.messages.create(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="anthropic-claude",
    max_tokens=100,
)
```

</TabItem>

<TabItem value="openai-proxy" label="OpenAI">

1. Setup config.yaml

```yaml
model_list:
    - model_name: openai-gpt4
      litellm_params:
        model: openai/gpt-4
        api_key: os.environ/OPENAI_API_KEY
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python showLineNumbers title="OpenAI Example using LiteLLM Proxy Server"
import anthropic

# point anthropic sdk to litellm proxy 
client = anthropic.Anthropic(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

response = client.messages.create(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="openai-gpt4",
    max_tokens=100,
)
```

</TabItem>

<TabItem value="gemini-proxy" label="Google AI Studio">

1. Setup config.yaml

```yaml
model_list:
    - model_name: gemini-2-flash
      litellm_params:
        model: gemini/gemini-2.0-flash-exp
        api_key: os.environ/GEMINI_API_KEY
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python showLineNumbers title="Google Gemini Example using LiteLLM Proxy Server"
import anthropic

# point anthropic sdk to litellm proxy 
client = anthropic.Anthropic(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

response = client.messages.create(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="gemini-2-flash",
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
        model: vertex_ai/gemini-2.0-flash-exp
        vertex_project: your-gcp-project-id
        vertex_location: us-central1
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python showLineNumbers title="Vertex AI Example using LiteLLM Proxy Server"
import anthropic

# point anthropic sdk to litellm proxy 
client = anthropic.Anthropic(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

response = client.messages.create(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="vertex-gemini",
    max_tokens=100,
)
```

</TabItem>

<TabItem value="bedrock-proxy" label="AWS Bedrock">

1. Setup config.yaml

```yaml
model_list:
    - model_name: bedrock-claude
      litellm_params:
        model: bedrock/anthropic.claude-3-sonnet-20240229-v1:0
        aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
        aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
        aws_region_name: us-west-2
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python showLineNumbers title="AWS Bedrock Example using LiteLLM Proxy Server"
import anthropic

# point anthropic sdk to litellm proxy 
client = anthropic.Anthropic(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

response = client.messages.create(
    messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
    model="bedrock-claude",
    max_tokens=100,
)
```

</TabItem>

<TabItem value="curl-proxy" label="curl">

```bash showLineNumbers title="Example using LiteLLM Proxy Server"
curl -L -X POST 'http://0.0.0.0:4000/v1/messages' \
-H 'content-type: application/json' \
-H 'x-api-key: $LITELLM_API_KEY' \
-H 'anthropic-version: 2023-06-01' \
-d '{
  "model": "anthropic-claude",
  "messages": [
    {
      "role": "user",
      "content": "Hello, can you tell me a short joke?"
    }
  ],
  "max_tokens": 100
}'
```

</TabItem>
</Tabs>

## Request Format
---

Request body will be in the Anthropic messages API format. **litellm follows the Anthropic messages specification for this endpoint.**

#### Example request body

```json
{
  "model": "claude-3-7-sonnet-20250219",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Hello, world"
    }
  ]
}
```

#### Required Fields
- **model** (string):  
  The model identifier (e.g., `"claude-3-7-sonnet-20250219"`).
- **max_tokens** (integer):  
  The maximum number of tokens to generate before stopping.  
  _Note: The model may stop before reaching this limit; value must be greater than 1._
- **messages** (array of objects):  
  An ordered list of conversational turns.  
  Each message object must include:
  - **role** (enum: `"user"` or `"assistant"`):  
    Specifies the speaker of the message.
  - **content** (string or array of content blocks):  
    The text or content blocks (e.g., an array containing objects with a `type` such as `"text"`) that form the message.  
    _Example equivalence:_
    ```json
    {"role": "user", "content": "Hello, Claude"}
    ```
    is equivalent to:
    ```json
    {"role": "user", "content": [{"type": "text", "text": "Hello, Claude"}]}
    ```

#### Optional Fields
- **metadata** (object):  
  Contains additional metadata about the request (e.g., `user_id` as an opaque identifier).
- **stop_sequences** (array of strings):  
  Custom sequences that, when encountered in the generated text, cause the model to stop.
- **stream** (boolean):  
  Indicates whether to stream the response using server-sent events.
- **system** (string or array):  
  A system prompt providing context or specific instructions to the model.
- **temperature** (number):  
  Controls randomness in the model's responses. Valid range: `0 < temperature < 1`.
- **thinking** (object):  
  Configuration for enabling extended thinking. If enabled, it includes:
  - **budget_tokens** (integer):  
    Minimum of 1024 tokens (and less than `max_tokens`).
  - **type** (enum):  
    E.g., `"enabled"`.
- **tool_choice** (object):  
  Instructs how the model should utilize any provided tools.
- **tools** (array of objects):  
  Definitions for tools available to the model. Each tool includes:
  - **name** (string):  
    The tool's name.
  - **description** (string):  
    A detailed description of the tool.
  - **input_schema** (object):  
    A JSON schema describing the expected input format for the tool.
- **top_k** (integer):  
  Limits sampling to the top K options.
- **top_p** (number):  
  Enables nucleus sampling with a cumulative probability cutoff. Valid range: `0 < top_p < 1`.


## Response Format
---

Responses will be in the Anthropic messages API format.

#### Example Response

```json
{
  "content": [
    {
      "text": "Hi! My name is Claude.",
      "type": "text"
    }
  ],
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "model": "claude-3-7-sonnet-20250219",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503,
    "cache_creation_input_tokens": 2095,
    "cache_read_input_tokens": 0
  }
}
```

#### Response fields

- **content** (array of objects):  
  Contains the generated content blocks from the model. Each block includes:
  - **type** (string):  
    Indicates the type of content (e.g., `"text"`, `"tool_use"`, `"thinking"`, or `"redacted_thinking"`).
  - **text** (string):  
    The generated text from the model.  
    _Note: Maximum length is 5,000,000 characters._
  - **citations** (array of objects or `null`):  
    Optional field providing citation details. Each citation includes:
    - **cited_text** (string):  
      The excerpt being cited.
    - **document_index** (integer):  
      An index referencing the cited document.
    - **document_title** (string or `null`):  
      The title of the cited document.
    - **start_char_index** (integer):  
      The starting character index for the citation.
    - **end_char_index** (integer):  
      The ending character index for the citation.
    - **type** (string):  
      Typically `"char_location"`.

- **id** (string):  
  A unique identifier for the response message.  
  _Note: The format and length of IDs may change over time._

- **model** (string):  
  Specifies the model that generated the response.

- **role** (string):  
  Indicates the role of the generated message. For responses, this is always `"assistant"`.

- **stop_reason** (string):  
  Explains why the model stopped generating text. Possible values include:
  - `"end_turn"`: The model reached a natural stopping point.
  - `"max_tokens"`: The generation stopped because the maximum token limit was reached.
  - `"stop_sequence"`: A custom stop sequence was encountered.
  - `"tool_use"`: The model invoked one or more tools.

- **stop_sequence** (string or `null`):  
  Contains the specific stop sequence that caused the generation to halt, if applicable; otherwise, it is `null`.

- **type** (string):  
  Denotes the type of response object, which is always `"message"`.

- **usage** (object):  
  Provides details on token usage for billing and rate limiting. This includes:
  - **input_tokens** (integer):  
    Total number of input tokens processed.
  - **output_tokens** (integer):  
    Total number of output tokens generated.
  - **cache_creation_input_tokens** (integer or `null`):  
    Number of tokens used to create a cache entry.
  - **cache_read_input_tokens** (integer or `null`):  
    Number of tokens read from the cache.
