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

Just replace `https://REGION-aiplatform.googleapis.com` with `LITELLM_PROXY_BASE_URL/vertex_ai`


#### **Example Usage**

<Tabs>
<TabItem value="curl" label="curl">

```bash
curl http://localhost:4000/vertex_ai/publishers/google/models/gemini-1.0-pro:generateContent \
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
curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.0-pro:generateContent \
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

1. Pass Vertex Credetials client side to proxy server

2. Set Vertex AI credentials on proxy server


## Usage Examples

### Gemini API (Generate Content)



```shell
curl http://localhost:4000/vertex_ai/publishers/google/models/gemini-1.5-flash-001:generateContent \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```



### Embeddings API


```shell
curl http://localhost:4000/vertex_ai/publishers/google/models/textembedding-gecko@001:predict \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{"instances":[{"content": "gm"}]}'
```


### Imagen API

```shell
curl http://localhost:4000/vertex_ai/publishers/google/models/imagen-3.0-generate-001:predict \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{"instances":[{"prompt": "make an otter"}], "parameters": {"sampleCount": 1}}'
```


### Count Tokens API

```shell
curl http://localhost:4000/vertex_ai/publishers/google/models/gemini-1.5-flash-001:countTokens \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```
### Tuning API 

Create Fine Tuning Job


```shell
curl http://localhost:4000/vertex_ai/tuningJobs \
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
curl http://localhost:4000/vertex_ai/publishers/google/models/gemini-1.0-pro:generateContent \
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
curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.0-pro:generateContent \
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