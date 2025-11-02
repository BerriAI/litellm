import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vertex AI Embedding

## Usage - Embedding

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
from litellm import embedding
litellm.vertex_project = "hardy-device-38811" # Your Project ID
litellm.vertex_location = "us-central1"  # proj location

response = embedding(
    model="vertex_ai/textembedding-gecko",
    input=["good morning from litellm"],
)
print(response)
```
</TabItem>

<TabItem value="proxy" label="LiteLLM PROXY">


1. Add model to config.yaml
```yaml
model_list:
  - model_name: snowflake-arctic-embed-m-long-1731622468876
    litellm_params:
      model: vertex_ai/<your-model-id>
      vertex_project: "adroit-crow-413218"
      vertex_location: "us-central1"
      vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request using OpenAI Python SDK, Langchain Python SDK

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    model="snowflake-arctic-embed-m-long-1731622468876", 
    input = ["good morning from litellm", "this is another item"],
)

print(response)
```


</TabItem>
</Tabs>

#### Supported Embedding Models
All models listed [here](https://github.com/BerriAI/litellm/blob/57f37f743886a0249f630a6792d49dffc2c5d9b7/model_prices_and_context_window.json#L835) are supported

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| text-embedding-004 | `embedding(model="vertex_ai/text-embedding-004", input)` | 
| text-multilingual-embedding-002 | `embedding(model="vertex_ai/text-multilingual-embedding-002", input)` | 
| textembedding-gecko | `embedding(model="vertex_ai/textembedding-gecko", input)` | 
| textembedding-gecko-multilingual | `embedding(model="vertex_ai/textembedding-gecko-multilingual", input)` | 
| textembedding-gecko-multilingual@001 | `embedding(model="vertex_ai/textembedding-gecko-multilingual@001", input)` | 
| textembedding-gecko@001 | `embedding(model="vertex_ai/textembedding-gecko@001", input)` | 
| textembedding-gecko@003 | `embedding(model="vertex_ai/textembedding-gecko@003", input)` | 
| text-embedding-preview-0409 | `embedding(model="vertex_ai/text-embedding-preview-0409", input)` |
| text-multilingual-embedding-preview-0409 | `embedding(model="vertex_ai/text-multilingual-embedding-preview-0409", input)` | 
| Fine-tuned OR Custom Embedding models | `embedding(model="vertex_ai/<your-model-id>", input)` | 

### Supported OpenAI (Unified) Params

| [param](../embedding/supported_embedding.md#input-params-for-litellmembedding) | type | [vertex equivalent](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api) |
|-------|-------------|--------------------|
| `input` | **string or List[string]** | `instances` |
| `dimensions` | **int** | `output_dimensionality` |
| `input_type` | **Literal["RETRIEVAL_QUERY","RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION"]** | `task_type` |

#### Usage with OpenAI (Unified) Params


<Tabs>
<TabItem value="sdk" label="SDK">

```python
response = litellm.embedding(
    model="vertex_ai/text-embedding-004",
    input=["good morning from litellm", "gm"]
    input_type = "RETRIEVAL_DOCUMENT",
    dimensions=1,
)
```
</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY">


```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    model="text-embedding-004", 
    input = ["good morning from litellm", "gm"],
    dimensions=1,
    extra_body = {
        "input_type": "RETRIEVAL_QUERY",
    }
)

print(response)
```
</TabItem>
</Tabs>


### Supported Vertex Specific Params

| param | type |
|-------|-------------|
| `auto_truncate` | **bool** |
| `task_type` | **Literal["RETRIEVAL_QUERY","RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION"]** |
| `title` | **str** |

#### Usage with Vertex Specific Params  (Use `task_type` and `title`)

You can pass any vertex specific params to the embedding model. Just pass them to the embedding function like this: 

[Relevant Vertex AI doc with all embedding params](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#request_body)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
response = litellm.embedding(
    model="vertex_ai/text-embedding-004",
    input=["good morning from litellm", "gm"]
    task_type = "RETRIEVAL_DOCUMENT",
    title = "test",
    dimensions=1,
    auto_truncate=True,
)
```
</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY">


```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    model="text-embedding-004", 
    input = ["good morning from litellm", "gm"],
    dimensions=1,
    extra_body = {
        "task_type": "RETRIEVAL_QUERY",
        "auto_truncate": True,
        "title": "test",
    }
)

print(response)
```
</TabItem>
</Tabs>

## **BGE Embeddings**

Use BGE (Baidu General Embedding) models deployed on Vertex AI.

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers title="Using BGE on Vertex AI"
import litellm

response = litellm.embedding(
    model="vertex_ai/bge/<your-endpoint-id>",
    input=["Hello", "World"],
    vertex_project="your-project-id",
    vertex_location="your-location"
)

print(response)
```

</TabItem>

<TabItem value="proxy" label="LiteLLM PROXY">

1. Add model to config.yaml
```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: bge-embedding
    litellm_params:
      model: vertex_ai/bge/<your-endpoint-id>
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: your-credentials.json

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```bash
$ litellm --config /path/to/config.yaml
```

3. Make Request using OpenAI Python SDK

```python showLineNumbers title="Making requests to BGE"
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    model="bge-embedding",
    input=["good morning from litellm", "this is another item"]
)

print(response)
```

Using a Private Service Connect (PSC) endpoint

```yaml showLineNumbers title="config.yaml (PSC)"
model_list:
  - model_name: bge-small-en-v1.5
    litellm_params:
      model: vertex_ai/bge/1234567890 
      api_base: http://10.96.32.8  # Your PSC IP
      vertex_project: my-project-id  #optional
      vertex_location: us-central1 #optional
```

</TabItem>
</Tabs>

## **Multi-Modal Embeddings**


Known Limitations:
- Only supports 1 image / video / image per request
- Only supports GCS or base64 encoded images / videos

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

Using GCS Images

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input="gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png" # will be sent as a gcs image
)
```

Using base 64 encoded images

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input="data:image/jpeg;base64,..." # will be sent as a base64 encoded image
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: multimodalembedding@001
    litellm_params:
      model: vertex_ai/multimodalembedding@001
      vertex_project: "adroit-crow-413218"
      vertex_location: "us-central1"
      vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request use OpenAI Python SDK, Langchain Python SDK


<Tabs>

<TabItem value="OpenAI SDK" label="OpenAI SDK">

Requests with GCS Image / Video URI

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png",
)

print(response)
```

Requests with base64 encoded images

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = "data:image/jpeg;base64,...",
)

print(response)
```

</TabItem>

<TabItem value="langchain" label="Langchain">

Requests with GCS Image / Video URI
```python
from langchain_openai import OpenAIEmbeddings

embeddings_models = "multimodalembedding@001"

embeddings = OpenAIEmbeddings(
    model="multimodalembedding@001",
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",  # type: ignore
)


query_result = embeddings.embed_query(
    "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
)
print(query_result)

```

Requests with base64 encoded images

```python
from langchain_openai import OpenAIEmbeddings

embeddings_models = "multimodalembedding@001"

embeddings = OpenAIEmbeddings(
    model="multimodalembedding@001",
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",  # type: ignore
)


query_result = embeddings.embed_query(
    "data:image/jpeg;base64,..."
)
print(query_result)

```

</TabItem>

</Tabs>
</TabItem>


<TabItem value="proxy-vtx" label="LiteLLM PROXY (Vertex SDK)">

1. Add model to config.yaml
```yaml
default_vertex_config:
  vertex_project: "adroit-crow-413218"
  vertex_location: "us-central1"
  vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request use OpenAI Python SDK

```python
import vertexai

from vertexai.vision_models import Image, MultiModalEmbeddingModel, Video
from vertexai.vision_models import VideoSegmentConfig
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
        headers['Authorization'] = f'Bearer {self.token}'

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
    credentials = credentials,
    api_transport="rest",
   
)

model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
image = Image.load_from_file(
    "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
)

embeddings = model.get_embeddings(
    image=image,
    contextual_text="Colosseum",
    dimension=1408,
)
print(f"Image Embedding: {embeddings.image_embedding}")
print(f"Text Embedding: {embeddings.text_embedding}")
```

</TabItem>
</Tabs>


### Text + Image + Video Embeddings

<Tabs>
<TabItem value="sdk" label="SDK">

Text + Image 

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input=["hey", "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"] # will be sent as a gcs image
)
```

Text + Video 

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input=["hey", "gs://my-bucket/embeddings/supermarket-video.mp4"] # will be sent as a gcs image
)
```

Image + Video 

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input=["gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png", "gs://my-bucket/embeddings/supermarket-video.mp4"] # will be sent as a gcs image
)
```


</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: multimodalembedding@001
    litellm_params:
      model: vertex_ai/multimodalembedding@001
      vertex_project: "adroit-crow-413218"
      vertex_location: "us-central1"
      vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request use OpenAI Python SDK, Langchain Python SDK


Text + Image 

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = ["hey", "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"],
)

print(response)
```

Text + Video 
```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = ["hey", "gs://my-bucket/embeddings/supermarket-video.mp4"],
)

print(response)
```

Image + Video 
```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = ["gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png", "gs://my-bucket/embeddings/supermarket-video.mp4"],
)

print(response)
```

</TabItem>
</Tabs>