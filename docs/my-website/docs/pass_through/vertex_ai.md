import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Vertex AI Endpoints

Use VertexAI SDK to call endpoints on LiteLLM Gateway (native provider format)

:::tip

Looking for the Unified API (OpenAI format) for VertexAI ? [Go here - using vertexAI with LiteLLM SDK or LiteLLM Proxy Server](../providers/vertex.md)

:::

## Supported API Endpoints

- Gemini API
- Embeddings API
- Imagen API
- Code Completion API
- Batch prediction API
- Tuning API
- CountTokens API

## Authentication to Vertex AI

LiteLLM Proxy Server supports two methods of authentication to Vertex AI:

1. Pass Vertex Credetials client side to proxy server

2. Set Vertex AI credentials on proxy server

## Quick Start Usage 

<Tabs>
<TabItem value="without_default_config" label="Pass Vertex Credetials client side to proxy server">


#### 1. Start litellm proxy

```shell
litellm --config /path/to/config.yaml
```

#### 2. Test it 

```python
import vertexai
from vertexai.preview.generative_models import GenerativeModel

LITE_LLM_ENDPOINT = "http://localhost:4000"

vertexai.init(
    project="<your-vertex-ai-project-id>", # enter your project id
    location="<your-vertex-ai-location>", # enter your region
    api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex-ai", # route on litellm
    api_transport="rest",
)

model = GenerativeModel(model_name="gemini-1.0-pro")
model.generate_content("hi")

```

</TabItem>
<TabItem value="with_default_config" label="Set Vertex AI Credentials on Proxy Server">



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
)

model = GenerativeModel("gemini-1.5-flash-001")

response = model.generate_content(
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
)

print(response.text)
```

</TabItem>
</Tabs>


## Usage Examples

### Gemini API (Generate Content)

<Tabs>
<TabItem value="client_side" label="Vertex Python SDK (client side vertex credentials)">

```python
import vertexai
from vertexai.generative_models import GenerativeModel

LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    api_transport="rest",
   
)

model = GenerativeModel("gemini-1.5-flash-001")

response = model.generate_content(
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
)

print(response.text)
```

</TabItem>
<TabItem value="py" label="Vertex Python SDK (litellm virtual keys client side)">

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
<TabItem value="client_side" label="Vertex Python SDK (client side vertex credentials)">


```python
from typing import List, Optional
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import vertexai
from vertexai.generative_models import GenerativeModel

LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"

import datetime

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    api_transport="rest",
)


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
<TabItem value="py" label="Vertex Python SDK (litellm virtual keys client side)">

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
)


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

<Tabs>

<TabItem value="client_side" label="Vertex Python SDK (client side vertex credentials)">


```python
from typing import List, Optional
from vertexai.preview.vision_models import ImageGenerationModel
import vertexai
from google.auth.credentials import Credentials

LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"

import datetime

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    api_transport="rest",
)

model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

images = model.generate_images(
    prompt=prompt,
    # Optional parameters
    number_of_images=1,
    language="en",
    # You can't use a seed value and watermark at the same time.
    # add_watermark=False,
    # seed=100,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

images[0].save(location=output_file, include_generation_parameters=False)

# Optional. View the generated image in a notebook.
# images[0].show()

print(f"Created output image using {len(images[0]._image_bytes)} bytes")

```
</TabItem>

<TabItem value="py" label="Vertex Python SDK (litellm virtual keys client side)">

```python
from typing import List, Optional
from vertexai.preview.vision_models import ImageGenerationModel
import vertexai
from google.auth.credentials import Credentials

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
)

model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

images = model.generate_images(
    prompt=prompt,
    # Optional parameters
    number_of_images=1,
    language="en",
    # You can't use a seed value and watermark at the same time.
    # add_watermark=False,
    # seed=100,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

images[0].save(location=output_file, include_generation_parameters=False)

# Optional. View the generated image in a notebook.
# images[0].show()

print(f"Created output image using {len(images[0]._image_bytes)} bytes")

```

</TabItem>

<TabItem value="curl" label="Curl">

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/imagen-3.0-generate-001:predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"instances":[{"prompt": "make an otter"}], "parameters": {"sampleCount": 1}}'
```

</TabItem>

</Tabs>

### Count Tokens API


<Tabs>

<TabItem value="client_side" label="Vertex Python SDK (client side vertex credentials)">


```python
from typing import List, Optional
from vertexai.generative_models import GenerativeModel
import vertexai

LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"

import datetime

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    api_transport="rest",
)


model = GenerativeModel("gemini-1.5-flash-001")

prompt = "Why is the sky blue?"

# Prompt tokens count
response = model.count_tokens(prompt)
print(f"Prompt Token Count: {response.total_tokens}")
print(f"Prompt Character Count: {response.total_billable_characters}")

# Send text to Gemini
response = model.generate_content(prompt)

# Response tokens count
usage_metadata = response.usage_metadata
print(f"Prompt Token Count: {usage_metadata.prompt_token_count}")
print(f"Candidates Token Count: {usage_metadata.candidates_token_count}")
print(f"Total Token Count: {usage_metadata.total_token_count}")
```

</TabItem>


<TabItem value="py" label="Vertex Python SDK (litellm virtual keys client side)">

```python
from typing import List, Optional
from vertexai.generative_models import GenerativeModel
import vertexai
from google.auth.credentials import Credentials

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
)


model = GenerativeModel("gemini-1.5-flash-001")

prompt = "Why is the sky blue?"

# Prompt tokens count
response = model.count_tokens(prompt)
print(f"Prompt Token Count: {response.total_tokens}")
print(f"Prompt Character Count: {response.total_billable_characters}")

# Send text to Gemini
response = model.generate_content(prompt)

# Response tokens count
usage_metadata = response.usage_metadata
print(f"Prompt Token Count: {usage_metadata.prompt_token_count}")
print(f"Candidates Token Count: {usage_metadata.candidates_token_count}")
print(f"Total Token Count: {usage_metadata.total_token_count}")
```

</TabItem>

<TabItem value="curl" label="Curl">



```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.5-flash-001:countTokens \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```

</TabItem>
</Tabs>

### Tuning API 

Create Fine Tuning Job

<Tabs>

<TabItem value="client_side" label="Vertex Python SDK (client side vertex credentials)">

```python
from typing import List, Optional
from vertexai.preview.tuning import sft
import vertexai

LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"


vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    api_transport="rest",
)


# TODO(developer): Update project
vertexai.init(project=PROJECT_ID, location="us-central1")

sft_tuning_job = sft.train(
    source_model="gemini-1.0-pro-002",
    train_dataset="gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl",
)

# Polling for job completion
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()

print(sft_tuning_job.tuned_model_name)
print(sft_tuning_job.tuned_model_endpoint_name)
print(sft_tuning_job.experiment)

```

</TabItem>

<TabItem value="py" label="Vertex Python SDK (litellm virtual keys client side)">

```python
from typing import List, Optional
from vertexai.preview.tuning import sft
import vertexai
from google.auth.credentials import Credentials

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
)


# TODO(developer): Update project
vertexai.init(project=PROJECT_ID, location="us-central1")

sft_tuning_job = sft.train(
    source_model="gemini-1.0-pro-002",
    train_dataset="gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl",
)

# Polling for job completion
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()

print(sft_tuning_job.tuned_model_name)
print(sft_tuning_job.tuned_model_endpoint_name)
print(sft_tuning_job.experiment)
```

</TabItem>

<TabItem value="curl" label="Curl">

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

</TabItem>

</Tabs>


### Context Caching

Use Vertex AI Context Caching

[**Relevant VertexAI Docs**](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview)

<Tabs>

<TabItem value="proxy" label="LiteLLM PROXY">

1. Add model to config.yaml
```yaml
model_list:
  # used for /chat/completions, /completions, /embeddings endpoints
  - model_name: gemini-1.5-pro-001
    litellm_params:
      model: vertex_ai/gemini-1.5-pro-001
      vertex_project: "project-id"
      vertex_location: "us-central1"
      vertex_credentials: "adroit-crow-413218-a956eef1a2a8.json" # Add path to service account.json

# used for the /cachedContent and vertexAI native endpoints
default_vertex_config:
  vertex_project: "adroit-crow-413218"
  vertex_location: "us-central1"
  vertex_credentials: "adroit-crow-413218-a956eef1a2a8.json" # Add path to service account.json

```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request!
We make the request in two steps:
- Create a cachedContents object
- Use the cachedContents object in your /chat/completions 

**Create a cachedContents object**

First, create a cachedContents object by calling the Vertex `cachedContents` endpoint. The LiteLLM proxy forwards the `/cachedContents` request to the VertexAI API.

```python
import httpx

# Set Litellm proxy variables
LITELLM_BASE_URL = "http://0.0.0.0:4000"
LITELLM_PROXY_API_KEY = "sk-1234"

httpx_client = httpx.Client(timeout=30)

print("Creating cached content")
create_cache = httpx_client.post(
    url=f"{LITELLM_BASE_URL}/vertex-ai/cachedContents",
    headers={"Authorization": f"Bearer {LITELLM_PROXY_API_KEY}"},
    json={
        "model": "gemini-1.5-pro-001",
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": "This is sample text to demonstrate explicit caching." * 4000
                }]
            }
        ],
    }
)

print("Response from create_cache:", create_cache)
create_cache_response = create_cache.json()
print("JSON from create_cache:", create_cache_response)
cached_content_name = create_cache_response["name"]
```

**Use the cachedContents object in your /chat/completions request to VertexAI**

```python
import openai

# Set Litellm proxy variables
LITELLM_BASE_URL = "http://0.0.0.0:4000"
LITELLM_PROXY_API_KEY = "sk-1234"

client = openai.OpenAI(api_key=LITELLM_PROXY_API_KEY, base_url=LITELLM_BASE_URL)

response = client.chat.completions.create(
    model="gemini-1.5-pro-001",
    max_tokens=8192,
    messages=[
        {
            "role": "user",
            "content": "What is the sample text about?",
        },
    ],
    temperature=0.7,
    extra_body={"cached_content": cached_content_name},  # Use the cached content
)

print("Response from proxy:", response)
```

</TabItem>
</Tabs>