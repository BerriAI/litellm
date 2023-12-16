import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Embeddings - `/embeddings`

See supported Embedding Providers & Models [here](https://docs.litellm.ai/docs/embedding/supported_embedding)


## Quick start
Here's how to route between GPT-J embedding (sagemaker endpoint), Amazon Titan embedding (Bedrock) and Azure OpenAI embedding on the proxy server: 

1. Set models in your config.yaml
```yaml
model_list:
  - model_name: sagemaker-embeddings
    litellm_params: 
      model: "sagemaker/berri-benchmarking-gpt-j-6b-fp16"
  - model_name: amazon-embeddings
    litellm_params:
      model: "bedrock/amazon.titan-embed-text-v1"
  - model_name: azure-embeddings
    litellm_params: 
      model: "azure/azure-embedding-model"
      api_base: "os.environ/AZURE_API_BASE" # os.getenv("AZURE_API_BASE")
      api_key: "os.environ/AZURE_API_KEY" # os.getenv("AZURE_API_KEY")
      api_version: "2023-07-01-preview"

general_settings:
  master_key: sk-1234 # [OPTIONAL] if set all calls to proxy will require either this key or a valid generated token
```

2. Start the proxy
```shell
$ litellm --config /path/to/config.yaml
```

3. Test the embedding call

```shell
curl --location 'http://0.0.0.0:8000/v1/embeddings' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "input": "The food was delicious and the waiter..",
    "model": "sagemaker-embeddings",
}'
```

## `/embeddings` Request Format
Input, Output and Exceptions are mapped to the OpenAI format for all supported models

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:8000/embeddings' \
  --header 'Content-Type: application/json' \
  --data ' {
  "model": "text-embedding-ada-002",
  "input": ["write a litellm poem"]
  }'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
from openai import OpenAI

# set base_url to your proxy server
# set api_key to send to proxy server
client = OpenAI(api_key="<proxy-api-key>", base_url="http://0.0.0.0:8000")

response = openai.embeddings.create(
    input=["hello from litellm"],
    model="text-embedding-ada-002"
)

print(response)

```
</TabItem>

<TabItem value="langchain-embedding" label="Langchain Embeddings">

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="sagemaker-embeddings", openai_api_base="http://0.0.0.0:8000", openai_api_key="temp-key")


text = "This is a test document."

query_result = embeddings.embed_query(text)

print(f"SAGEMAKER EMBEDDINGS")
print(query_result[:5])

embeddings = OpenAIEmbeddings(model="bedrock-embeddings", openai_api_base="http://0.0.0.0:8000", openai_api_key="temp-key")

text = "This is a test document."

query_result = embeddings.embed_query(text)

print(f"BEDROCK EMBEDDINGS")
print(query_result[:5])

embeddings = OpenAIEmbeddings(model="bedrock-titan-embeddings", openai_api_base="http://0.0.0.0:8000", openai_api_key="temp-key")

text = "This is a test document."

query_result = embeddings.embed_query(text)

print(f"TITAN EMBEDDINGS")
print(query_result[:5])
```
</TabItem>
</Tabs>



## `/embeddings` Response Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.0023064255,
        -0.009327292,
        .... 
        -0.0028842222,
      ],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}

```

## Supported Models

See supported Embedding Providers & Models [here](https://docs.litellm.ai/docs/embedding/supported_embedding)

#### Create Config.yaml

<Tabs>
<TabItem value="Hugging Face emb" label="Hugging Face Embeddings">
LiteLLM Proxy supports all <a href="https://huggingface.co/models?pipeline_tag=feature-extraction">Feature-Extraction Embedding models</a>.

```yaml
model_list:
  - model_name: deployed-codebert-base
    litellm_params: 
      # send request to deployed hugging face inference endpoint
      model: huggingface/microsoft/codebert-base # add huggingface prefix so it routes to hugging face
      api_key: hf_LdS                            # api key for hugging face inference endpoint
      api_base: https://uysneno1wv2wd4lw.us-east-1.aws.endpoints.huggingface.cloud # your hf inference endpoint 
  - model_name: codebert-base
    litellm_params: 
      # no api_base set, sends request to hugging face free inference api https://api-inference.huggingface.co/models/
      model: huggingface/microsoft/codebert-base # add huggingface prefix so it routes to hugging face
      api_key: hf_LdS                            # api key for hugging face                     

```

</TabItem>

<TabItem value="azure" label="Azure OpenAI Embeddings">

```yaml
model_list:
  - model_name: azure-embedding-model # model group
    litellm_params:
      model: azure/azure-embedding-model # model name for litellm.embedding(model=azure/azure-embedding-model) call
      api_base: your-azure-api-base
      api_key: your-api-key
      api_version: 2023-07-01-preview
```

</TabItem>

<TabItem value="openai" label="OpenAI Embeddings">

```yaml
model_list:
- model_name: text-embedding-ada-002 # model group
  litellm_params:
    model: text-embedding-ada-002 # model name for litellm.embedding(model=text-embedding-ada-002) 
    api_key: your-api-key-1
- model_name: text-embedding-ada-002 
  litellm_params:
    model: text-embedding-ada-002
    api_key: your-api-key-2
```

</TabItem>

<TabItem value="openai emb" label="OpenAI Compatible Embeddings">

<p>Use this for calling <a href="https://github.com/xorbitsai/inference">/embedding endpoints on OpenAI Compatible Servers</a>.</p>

**Note add `openai/` prefix to `litellm_params`: `model` so litellm knows to route to OpenAI**

```yaml
model_list:
- model_name: text-embedding-ada-002  # model group
  litellm_params:
    model: openai/<your-model-name>   # model name for litellm.embedding(model=text-embedding-ada-002) 
    api_base: <model-api-base>
```

</TabItem>
</Tabs>

#### Start Proxy
```shell
litellm --config config.yaml
```

#### Make Request
Sends Request to `deployed-codebert-base`

```shell
curl --location 'http://0.0.0.0:8000/embeddings' \
  --header 'Content-Type: application/json' \
  --data ' {
  "model": "deployed-codebert-base",
  "input": ["write a litellm poem"]
  }'
```





