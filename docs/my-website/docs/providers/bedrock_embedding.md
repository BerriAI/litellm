# Bedrock Embedding

## Supported Embedding Models

| Provider | LiteLLM Route | AWS Documentation |
|----------|---------------|-------------------|
| Amazon Titan | `bedrock/amazon.*` | [Amazon Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) |
| Cohere | `bedrock/cohere.*` | [Cohere Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-embed.html) |
| TwelveLabs | `bedrock/us.twelvelabs.*` | [TwelveLabs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-twelvelabs.html) |

### API keys
This can be set as env variables or passed as **params to litellm.embedding()**
```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = ""        # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = ""    # Secret access key
os.environ["AWS_REGION_NAME"] = ""           # us-east-1, us-east-2, us-west-1, us-west-2
```

## Usage
### LiteLLM Python SDK
```python
from litellm import embedding
response = embedding(
    model="bedrock/amazon.titan-embed-text-v1",
    input=["good morning from litellm"],
)
print(response)
```

### LiteLLM Proxy Server

#### 1. Setup config.yaml
```yaml
model_list:
  - model_name: titan-embed-v1
    litellm_params:
      model: bedrock/amazon.titan-embed-text-v1
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
  - model_name: titan-embed-v2
    litellm_params:
      model: bedrock/amazon.titan-embed-text-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
```

#### 2. Start Proxy 
```bash
litellm --config /path/to/config.yaml
```

#### 3. Use with OpenAI Python SDK
```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.embeddings.create(
    input=["good morning from litellm"],
    model="titan-embed-v1"
)
print(response)
```

#### 4. Use with LiteLLM Python SDK
```python
import litellm
response = litellm.embedding(
    model="titan-embed-v1", # model alias from config.yaml
    input=["good morning from litellm"],
    api_base="http://0.0.0.0:4000",
    api_key="anything"
)
print(response)
```

## Supported AWS Bedrock Embedding Models

| Model Name           | Usage                               | Supported Additional OpenAI params |
|----------------------|---------------------------------------------|-----|
| Titan Embeddings V2 | `embedding(model="bedrock/amazon.titan-embed-text-v2:0", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/amazon_titan_v2_transformation.py#L59) |
| Titan Embeddings - V1 | `embedding(model="bedrock/amazon.titan-embed-text-v1", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/amazon_titan_g1_transformation.py#L53)
| Titan Multimodal Embeddings | `embedding(model="bedrock/amazon.titan-embed-image-v1", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/amazon_titan_multimodal_transformation.py#L28) |
| TwelveLabs Marengo Embed 2.7 | `embedding(model="bedrock/us.twelvelabs.marengo-embed-2-7-v1:0", input=input)` | Supports multimodal input (text, video, audio, image) |
| Cohere Embeddings - English | `embedding(model="bedrock/cohere.embed-english-v3", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/cohere_transformation.py#L18)
| Cohere Embeddings - Multilingual | `embedding(model="bedrock/cohere.embed-multilingual-v3", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/cohere_transformation.py#L18)

### Advanced - [Drop Unsupported Params](https://docs.litellm.ai/docs/completion/drop_params#openai-proxy-usage)

### Advanced - [Pass model/provider-specific Params](https://docs.litellm.ai/docs/completion/provider_specific_params#proxy-usage)