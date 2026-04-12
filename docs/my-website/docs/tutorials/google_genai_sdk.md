import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Google GenAI SDK with LiteLLM

Use Google's official GenAI SDK (JavaScript/TypeScript and Python) with any LLM provider through LiteLLM Proxy.

The Google GenAI SDK (`@google/genai` for JS, `google-genai` for Python) provides a native interface for calling Gemini models. By pointing it to LiteLLM, you can use the same SDK with OpenAI, Anthropic, Bedrock, Azure, Vertex AI, or any other provider — while keeping the native Gemini request/response format.

## Why Use LiteLLM with Google GenAI SDK?

**Developer Benefits:**
- **Universal Model Access**: Use any LiteLLM-supported model (Anthropic, OpenAI, Vertex AI, Bedrock, etc.) through the Google GenAI SDK interface
- **Higher Rate Limits & Reliability**: Load balance across multiple models and providers to avoid hitting individual provider limits, with fallbacks to ensure you get responses even if one provider fails

**Proxy Admin Benefits:**
- **Centralized Management**: Control access to all models through a single LiteLLM proxy instance without giving developers API keys to each provider
- **Budget Controls**: Set spending limits and track costs across all SDK usage
- **Logging & Observability**: Track all requests with cost tracking, logging, and analytics

| Feature | Supported | Notes |
|---------|-----------|-------|
| Cost Tracking | ✅ | All models on `/generateContent` endpoint |
| Logging | ✅ | Works across all integrations |
| Streaming | ✅ | `streamGenerateContent` supported |
| Virtual Keys | ✅ | Use LiteLLM keys instead of Google keys |
| Load Balancing | ✅ | Via native router endpoints |
| Fallbacks | ✅ | Via native router endpoints |

## Quick Start

### 1. Install the SDK

<Tabs>
<TabItem value="js" label="JavaScript/TypeScript">

```bash
npm install @google/genai
```

</TabItem>
<TabItem value="python" label="Python">

```bash
pip install google-genai
```

</TabItem>
</Tabs>

### 2. Start LiteLLM Proxy

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gemini-2.5-flash
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY
```

```bash
litellm --config config.yaml
```

### 3. Call the SDK through LiteLLM

<Tabs>
<TabItem value="js" label="JavaScript/TypeScript">

```javascript title="index.js" showLineNumbers
const { GoogleGenAI } = require("@google/genai");

const ai = new GoogleGenAI({
  apiKey: "sk-1234",  // LiteLLM virtual key (not a Google key)
  httpOptions: {
    baseUrl: "http://localhost:4000/gemini",  // LiteLLM proxy URL
  },
});

async function main() {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "Explain how AI works",
  });
  console.log(response.text);
}

main();
```

</TabItem>
<TabItem value="python" label="Python">

```python title="main.py" showLineNumbers
from google import genai

client = genai.Client(
    api_key="sk-1234",  # LiteLLM virtual key (not a Google key)
    http_options={"base_url": "http://localhost:4000/gemini"},  # LiteLLM proxy URL
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works",
)
print(response.text)
```

</TabItem>
<TabItem value="curl" label="curl">

```bash
curl "http://localhost:4000/gemini/v1beta/models/gemini-2.5-flash:generateContent?key=sk-1234" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{
    "contents": [{
      "parts": [{"text": "Explain how AI works"}]
    }]
  }'
```

</TabItem>
</Tabs>

## Streaming

<Tabs>
<TabItem value="js" label="JavaScript/TypeScript">

```javascript title="streaming.js" showLineNumbers
const { GoogleGenAI } = require("@google/genai");

const ai = new GoogleGenAI({
  apiKey: "sk-1234",
  httpOptions: {
    baseUrl: "http://localhost:4000/gemini",
  },
});

async function main() {
  const response = await ai.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Write a short poem about the ocean",
  });

  for await (const chunk of response) {
    process.stdout.write(chunk.text);
  }
}

main();
```

</TabItem>
<TabItem value="python" label="Python">

```python title="streaming.py" showLineNumbers
from google import genai

client = genai.Client(
    api_key="sk-1234",
    http_options={"base_url": "http://localhost:4000/gemini"},
)

response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Write a short poem about the ocean",
)

for chunk in response:
    print(chunk.text, end="")
```

</TabItem>
</Tabs>

## Multi-turn Chat

<Tabs>
<TabItem value="js" label="JavaScript/TypeScript">

```javascript title="chat.js" showLineNumbers
const { GoogleGenAI } = require("@google/genai");

const ai = new GoogleGenAI({
  apiKey: "sk-1234",
  httpOptions: {
    baseUrl: "http://localhost:4000/gemini",
  },
});

async function main() {
  const chat = ai.chats.create({
    model: "gemini-2.5-flash",
  });

  const response1 = await chat.sendMessage({ message: "I have 2 dogs and 3 cats." });
  console.log(response1.text);

  const response2 = await chat.sendMessage({ message: "How many pets is that in total?" });
  console.log(response2.text);
}

main();
```

</TabItem>
<TabItem value="python" label="Python">

```python title="chat.py" showLineNumbers
from google import genai

client = genai.Client(
    api_key="sk-1234",
    http_options={"base_url": "http://localhost:4000/gemini"},
)

chat = client.chats.create(model="gemini-2.5-flash")

response1 = chat.send_message("I have 2 dogs and 3 cats.")
print(response1.text)

response2 = chat.send_message("How many pets is that in total?")
print(response2.text)
```

</TabItem>
</Tabs>


## Advanced: Use Any Model with the GenAI SDK

By default, the GenAI SDK talks to Gemini models. But with LiteLLM's router, you can route GenAI SDK requests to **any provider** — Anthropic, OpenAI, Bedrock, etc.

This works by using `model_group_alias` to map Gemini model names to your desired provider models. LiteLLM handles the format translation internally.

:::info

For this to work, point the SDK `baseUrl` to `http://localhost:4000` (without `/gemini`). This routes requests through LiteLLM's native Google endpoints, which go through the router and support model aliasing.

:::

<Tabs>
<TabItem value="anthropic" label="Anthropic">

Route `gemini-2.5-flash` requests to Claude Sonnet:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  model_group_alias: {"gemini-2.5-flash": "claude-sonnet"}
```

</TabItem>
<TabItem value="openai" label="OpenAI">

Route `gemini-2.5-flash` requests to GPT-4o:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o-model
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  model_group_alias: {"gemini-2.5-flash": "gpt-4o-model"}
```

</TabItem>
<TabItem value="bedrock" label="Bedrock">

Route `gemini-2.5-flash` requests to Claude on Bedrock:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: bedrock-claude
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

router_settings:
  model_group_alias: {"gemini-2.5-flash": "bedrock-claude"}
```

</TabItem>
<TabItem value="multi" label="Multi-Provider Load Balancing">

Load balance across Anthropic and OpenAI:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: my-model
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: my-model
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  model_group_alias: {"gemini-2.5-flash": "my-model"}
```

</TabItem>
</Tabs>

Then use the SDK with `baseUrl` pointing to LiteLLM (without `/gemini`):

<Tabs>
<TabItem value="js" label="JavaScript/TypeScript">

```javascript title="any_model.js" showLineNumbers
const { GoogleGenAI } = require("@google/genai");

const ai = new GoogleGenAI({
  apiKey: "sk-1234",
  httpOptions: {
    baseUrl: "http://localhost:4000",  // No /gemini — goes through the router
  },
});

async function main() {
  // This calls Claude/GPT-4o/Bedrock under the hood via model_group_alias
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "Hello from any model!",
  });
  console.log(response.text);
}

main();
```

</TabItem>
<TabItem value="python" label="Python">

```python title="any_model.py" showLineNumbers
from google import genai

client = genai.Client(
    api_key="sk-1234",
    http_options={"base_url": "http://localhost:4000"},  # No /gemini
)

# This calls Claude/GPT-4o/Bedrock under the hood via model_group_alias
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello from any model!",
)
print(response.text)
```

</TabItem>
</Tabs>


## Pass-through vs Native Router Endpoints

LiteLLM offers two ways to handle GenAI SDK requests:

| | Pass-through (`/gemini`) | Native Router (`/`) |
|---|---|---|
| **baseUrl** | `http://localhost:4000/gemini` | `http://localhost:4000` |
| **Models** | Gemini only | Any provider via `model_group_alias` |
| **Translation** | None — proxies directly to Google | Translates internally |
| **Cost Tracking** | ✅ | ✅ |
| **Virtual Keys** | ✅ | ✅ |
| **Load Balancing** | ❌ | ✅ |
| **Fallbacks** | ❌ | ✅ |
| **Best for** | Simple Gemini proxy | Multi-provider routing |

## Environment Variable Configuration

You can also configure the SDK via environment variables instead of code:

```bash
# For JavaScript SDK (@google/genai)
export GOOGLE_GEMINI_BASE_URL="http://localhost:4000/gemini"
export GEMINI_API_KEY="sk-1234"

# For Python SDK (google-genai)
# Note: The Python SDK does not support a base URL env var.
# Configure it in code with http_options={"base_url": "..."} instead.
export GEMINI_API_KEY="sk-1234"
```

This is especially useful for tools built on top of the GenAI SDK (like [Gemini CLI](./litellm_gemini_cli.md)).

## Related Resources

- [Gemini CLI with LiteLLM](./litellm_gemini_cli.md)
- [Google AI Studio Pass-Through](../pass_through/google_ai_studio)
- [Google ADK with LiteLLM](./google_adk.md)
- [LiteLLM Proxy Quick Start](../proxy/quick_start)
- [`@google/genai` npm package](https://www.npmjs.com/package/@google/genai)
- [`google-genai` PyPI package](https://pypi.org/project/google-genai/)
