import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vertex AI SDK

Pass-through endpoints for Vertex AI - call provider-specific endpoint, in native format (no translation).

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ | supports all models on `/generateContent` endpoint |
| Logging | ✅ | works across all integrations |
| End-user Tracking | ❌ | [Tell us if you need this](https://github.com/BerriAI/litellm/issues/new) |
| Streaming | ✅ | |

## Supported Endpoints

LiteLLM supports 3 vertex ai passthrough routes:

1. `/vertex_ai` → routes to `https://{vertex_location}-aiplatform.googleapis.com/`
2. `/vertex_ai/discovery` → routes to [`https://discoveryengine.googleapis.com`](https://discoveryengine.googleapis.com/)
3. `/vertex_ai/live` → upgrades to the Vertex AI Live API WebSocket (`google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent`)

## How to use

Just replace `https://REGION-aiplatform.googleapis.com` with `LITELLM_PROXY_BASE_URL/vertex_ai`

LiteLLM supports 3 flows for calling Vertex AI endpoints via pass-through:

1. **Specific Credentials**: Admin sets passthrough credentials for a specific project/region.

2. **Default Credentials**: Admin sets default credentials.

3. **Client-Side Credentials**: User can send client-side credentials through to Vertex AI (default behavior - if no default or mapped credentials are found, the request is passed through directly).


## Example Usage

<Tabs>
<TabItem value="specific_credentials" label="Specific Project/Region">

```yaml
model_list:
  - model_name: gemini-1.0-pro
    litellm_params:
      model: vertex_ai/gemini-1.0-pro
      vertex_project: adroit-crow-413218
      vertex_region: us-central1
      vertex_credentials: /path/to/credentials.json
      use_in_pass_through: true # 👈 KEY CHANGE
```

</TabItem>
<TabItem value="default_credentials" label="Default Credentials">

<Tabs>
<TabItem value="yaml" label="Set in config.yaml">

```yaml
default_vertex_config: 
  vertex_project: adroit-crow-413218
  vertex_region: us-central1
  vertex_credentials: /path/to/credentials.json
```
</TabItem>
<TabItem value="env_var" label="Set in environment variables">

```bash
export DEFAULT_VERTEXAI_PROJECT="adroit-crow-413218"
export DEFAULT_VERTEXAI_LOCATION="us-central1"
export DEFAULT_GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

</TabItem>
</Tabs>
</TabItem>
<TabItem value="client_credentials" label="Client Credentials">

Try Gemini 2.0 Flash (curl)

```
MODEL_ID="gemini-2.0-flash-001"
PROJECT_ID="YOUR_PROJECT_ID"
```

```bash
curl \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  -H "Content-Type: application/json" \
  "${LITELLM_PROXY_BASE_URL}/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/${MODEL_ID}:streamGenerateContent" -d \
  $'{
    "contents": {
      "role": "user",
      "parts": [
        {
        "fileData": {
          "mimeType": "image/png",
          "fileUri": "gs://generativeai-downloads/images/scones.jpg"
          }
        },
        {
          "text": "Describe this picture."
        }
      ]
    }
  }'
```

</TabItem>
</Tabs>


#### **Example Usage**

<Tabs>
<TabItem value="curl" label="curl">

```bash
curl http://localhost:4000/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/${MODEL_ID}:generateContent \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{
    "contents":[{
      "role": "user", 
      "parts":[{"text": "How are you doing today?"}]
    }]
  }'
```

</TabItem>
<TabItem value="js" label="Vertex Node.js SDK">

```javascript
const { VertexAI } = require('@google-cloud/vertexai');

const vertexAI = new VertexAI({
    project: 'your-project-id', // enter your vertex project id
    location: 'us-central1', // enter your vertex region
    apiEndpoint: "localhost:4000/vertex_ai" // <proxy-server-url>/vertex_ai # note, do not include 'https://' in the url
});

const model = vertexAI.getGenerativeModel({
    model: 'gemini-1.0-pro'
}, {
    customHeaders: {
        "x-litellm-api-key": "sk-1234" // Your litellm Virtual Key
    }
});

async function generateContent() {
    try {
        const prompt = {
            contents: [{
                role: 'user',
                parts: [{ text: 'How are you doing today?' }]
            }]
        };

        const response = await model.generateContent(prompt);
        console.log('Response:', response);
    } catch (error) {
        console.error('Error:', error);
    }
}

generateContent();
```

</TabItem>
</Tabs>


## Vertex AI Live API WebSocket

LiteLLM can now proxy the Vertex AI Live API to help you experiment with streaming audio/text from Gemini Live models without exposing Google credentials to clients.

- Configure default Vertex credentials via `default_vertex_config` or environment variables (see examples above).
- Connect to `wss://<PROXY_URL>/vertex_ai/live`. LiteLLM will exchange your saved credentials for a short-lived access token and forward messages bidirectionally.
- Optional query params `vertex_project`, `vertex_location`, and `model` let you override defaults for multi-project setups or global-only models.

```python title="client.py"
import asyncio
import json

from websockets.asyncio.client import connect


async def main() -> None:
    headers = {
        "x-litellm-api-key": "Bearer sk-your-litellm-key",
        "Content-Type": "application/json",
    }
    async with connect(
        "ws://localhost:4000/vertex_ai/live",
        additional_headers=headers,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "setup": {
                        "model": "projects/your-project/locations/us-central1/publishers/google/models/gemini-2.0-flash-live-preview-04-09",
                        "generation_config": {"response_modalities": ["TEXT"]},
                    }
                }
            )
        )

        async for message in ws:
            print("server:", message)


if __name__ == "__main__":
    asyncio.run(main())
```


## Quick Start

Let's call the Vertex AI [`/generateContent` endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference)

1. Add Vertex AI Credentials to your environment 

```bash
export DEFAULT_VERTEXAI_PROJECT="" # "adroit-crow-413218"
export DEFAULT_VERTEXAI_LOCATION="" # "us-central1"
export DEFAULT_GOOGLE_APPLICATION_CREDENTIALS="" # "/Users/Downloads/adroit-crow-413218-a956eef1a2a8.json"
```

2. Start LiteLLM Proxy 

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

Let's call the Google AI Studio token counting endpoint

```bash
curl http://localhost:4000/vertex-ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-1.0-pro:generateContent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "contents":[{
      "role": "user",
      "parts":[{"text": "How are you doing today?"}]
    }]
  }'
```



## Supported API Endpoints

- Gemini API
- Embeddings API
- Imagen API
- Code Completion API
- Batch prediction API
- Tuning API
- CountTokens API

#### Authentication to Vertex AI

LiteLLM Proxy Server supports two methods of authentication to Vertex AI:

1. Pass Vertex Credentials client side to proxy server

2. Set Vertex AI credentials on proxy server


## Usage Examples

### Gemini API (Generate Content)



```shell
curl http://localhost:4000/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-1.5-flash-001:generateContent \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```



### Embeddings API


```shell
curl http://localhost:4000/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/textembedding-gecko@001:predict \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{"instances":[{"content": "gm"}]}'
```


### Imagen API

```shell
curl http://localhost:4000/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/imagen-3.0-generate-001:predict \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{"instances":[{"prompt": "make an otter"}], "parameters": {"sampleCount": 1}}'
```


### Count Tokens API

```shell
curl http://localhost:4000/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-1.5-flash-001:countTokens \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```
### Tuning API 

Create Fine Tuning Job


```shell
curl http://localhost:4000/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-1.5-flash-001:tuningJobs \
      -H "Content-Type: application/json" \
      -H "x-litellm-api-key: Bearer sk-1234" \
      -d '{
  "baseModel": "gemini-1.0-pro-002",
  "supervisedTuningSpec" : {
      "training_dataset_uri": "gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl"
  }
}'
```

## Advanced

Pre-requisites
- [Setup proxy with DB](../proxy/virtual_keys.md#setup)

Use this, to avoid giving developers the raw Anthropic API key, but still letting them use Anthropic endpoints.

### Use with Virtual Keys 

1. Setup environment

```bash
export DATABASE_URL=""
export LITELLM_MASTER_KEY=""

# vertex ai credentials
export DEFAULT_VERTEXAI_PROJECT="" # "adroit-crow-413218"
export DEFAULT_VERTEXAI_LOCATION="" # "us-central1"
export DEFAULT_GOOGLE_APPLICATION_CREDENTIALS="" # "/Users/Downloads/adroit-crow-413218-a956eef1a2a8.json"
```

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

2. Generate virtual key 

```bash
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'x-litellm-api-key: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{}'
```

Expected Response 

```bash
{
    ...
    "key": "sk-1234ewknldferwedojwojw"
}
```

3. Test it! 


```bash
curl http://localhost:4000/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-1.0-pro:generateContent \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{
    "contents":[{
      "role": "user", 
      "parts":[{"text": "How are you doing today?"}]
    }]
  }'
```

### Send `tags` in request headers

Use this if you wants `tags` to be tracked in the LiteLLM DB and on logging callbacks

Pass `tags` in request headers as a comma separated list. In the example below the following tags will be tracked 

```
tags: ["vertex-js-sdk", "pass-through-endpoint"]
```

<Tabs>
<TabItem value="curl" label="curl">

```bash
curl http://localhost:4000/vertex_ai/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-1.0-pro:generateContent \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -H "tags: vertex-js-sdk,pass-through-endpoint" \
  -d '{
    "contents":[{
      "role": "user", 
      "parts":[{"text": "How are you doing today?"}]
    }]
  }'
```

</TabItem>
<TabItem value="js" label="Vertex Node.js SDK">

```javascript
const { VertexAI } = require('@google-cloud/vertexai');

const vertexAI = new VertexAI({
    project: 'your-project-id', // enter your vertex project id
    location: 'us-central1', // enter your vertex region
    apiEndpoint: "localhost:4000/vertex_ai" // <proxy-server-url>/vertex_ai # note, do not include 'https://' in the url
});

const model = vertexAI.getGenerativeModel({
    model: 'gemini-1.0-pro'
}, {
    customHeaders: {
        "x-litellm-api-key": "sk-1234", // Your litellm Virtual Key
        "tags": "vertex-js-sdk,pass-through-endpoint"
    }
});

async function generateContent() {
    try {
        const prompt = {
            contents: [{
                role: 'user',
                parts: [{ text: 'How are you doing today?' }]
            }]
        };

        const response = await model.generateContent(prompt);
        console.log('Response:', response);
    } catch (error) {
        console.error('Error:', error);
    }
}

generateContent();
```

</TabItem>
</Tabs>
