import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Vertex AI Endpoints (Pass-Through)

Pass-through endpoints for Vertex AI - call provider-specific endpoint, in native format (no translation).

:::tip

Looking for the Unified API (OpenAI format) for VertexAI ? [Go here - using vertexAI with LiteLLM SDK or LiteLLM Proxy Server](../docs/providers/vertex.md)

:::

## Supported API Endpoints

- Gemini API
- Embeddings API
- Imagen API
- Code Completion API
- Batch prediction API
- Tuning API
- CountTokens API

## Quick Start Usage 

#### 1. Set `default_vertex_config` on your `config.yaml`


Add the following credentials to your litellm config.yaml to use the Vertex AI endpoints.

```yaml
default_vertex_config:
  vertex_project: "adroit-crow-413218"
  vertex_location: "us-central1"
  vertex_credentials: "/Users/ishaanjaffer/Downloads/adroit-crow-413218-a956eef1a2a8.json" # Add path to service account.json
```

#### 2. Start litellm proxy

```shell
litellm --config /path/to/config.yaml
```

#### 3. Test it 

```python
import vertexai
from google.auth.credentials import Credentials
from vertexai.generative_models import GenerativeModel

LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"

import datetime


class CredentialsWrapper(Credentials):
    def __init__(self, token=None):
        super().__init__()
        self.token = token
        self.expiry = None  # or set to a future date if needed

    def refresh(self, request):
        pass

    def apply(self, headers, token=None):
        headers["Authorization"] = f"Bearer {self.token}"

    @property
    def expired(self):
        return False  # Always consider the token as non-expired

    @property
    def valid(self):
        return True  # Always consider the credentials as valid


credentials = CredentialsWrapper(token=LITELLM_PROXY_API_KEY)

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    credentials=credentials,
    api_transport="rest",
    request_metadata=[("Authorization", f"Bearer {LITELLM_PROXY_API_KEY}")],
)

model = GenerativeModel("gemini-1.5-flash-001")

response = model.generate_content(
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
)

print(response.text)
```

## Usage Examples

### Gemini API (Generate Content)

<Tabs>
<TabItem value="py" label="Vertex Python SDK">

```python
import vertexai
from google.auth.credentials import Credentials
from vertexai.generative_models import GenerativeModel

LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"

import datetime


class CredentialsWrapper(Credentials):
    def __init__(self, token=None):
        super().__init__()
        self.token = token
        self.expiry = None  # or set to a future date if needed

    def refresh(self, request):
        pass

    def apply(self, headers, token=None):
        headers["Authorization"] = f"Bearer {self.token}"

    @property
    def expired(self):
        return False  # Always consider the token as non-expired

    @property
    def valid(self):
        return True  # Always consider the credentials as valid


credentials = CredentialsWrapper(token=LITELLM_PROXY_API_KEY)

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    credentials=credentials,
    api_transport="rest",
    request_metadata=[("Authorization", f"Bearer {LITELLM_PROXY_API_KEY}")],
)

model = GenerativeModel("gemini-1.5-flash-001")

response = model.generate_content(
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
)

print(response.text)
```

</TabItem>
<TabItem value="Curl" label="Curl">

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.5-flash-001:generateContent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```

</TabItem>
</Tabs>


### Embeddings API

<Tabs>
<TabItem value="py" label="Vertex Python SDK">

```python
from typing import List, Optional
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import vertexai
from google.auth.credentials import Credentials
from vertexai.generative_models import GenerativeModel

LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"

import datetime


class CredentialsWrapper(Credentials):
    def __init__(self, token=None):
        super().__init__()
        self.token = token
        self.expiry = None  # or set to a future date if needed

    def refresh(self, request):
        pass

    def apply(self, headers, token=None):
        headers["Authorization"] = f"Bearer {self.token}"

    @property
    def expired(self):
        return False  # Always consider the token as non-expired

    @property
    def valid(self):
        return True  # Always consider the credentials as valid


credentials = CredentialsWrapper(token=LITELLM_PROXY_API_KEY)

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    credentials=credentials,
    api_transport="rest",
    request_metadata=[("Authorization", f"Bearer {LITELLM_PROXY_API_KEY}")],


def embed_text(
    texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "text-embedding-004",
    dimensionality: Optional[int] = 256,
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]
```
</TabItem>

<TabItem value="curl" label="Curl">

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/textembedding-gecko@001:predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"instances":[{"content": "gm"}]}'
```

</TabItem>

</Tabs>

### Imagen API

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/imagen-3.0-generate-001:predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"instances":[{"prompt": "make an otter"}], "parameters": {"sampleCount": 1}}'
```

### Count Tokens API

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.5-flash-001:countTokens \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```

### Tuning API 

Create Fine Tuning Job

```shell
curl http://localhost:4000/vertex-ai/tuningJobs \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer sk-1234" \
      -d '{
  "baseModel": "gemini-1.0-pro-002",
  "supervisedTuningSpec" : {
      "training_dataset_uri": "gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl"
  }
}'
```