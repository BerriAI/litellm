
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Cohere

## API KEYS

```python
import os 
os.environ["COHERE_API_KEY"] = ""
```

## Usage

### LiteLLM Python SDK

```python showLineNumbers
from litellm import completion

## set ENV variables
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = completion(
    model="command-r", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

#### Streaming

```python showLineNumbers
from litellm import completion

## set ENV variables
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = completion(
    model="command-r", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)

for chunk in response:
    print(chunk)
```



## Usage with LiteLLM Proxy 

Here's how to call Cohere with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export COHERE_API_KEY="your-api-key"
```

### 2. Start the proxy 

Define the cohere models you want to use in the config.yaml

```yaml showLineNumbers
model_list:
  - model_name: command-a-03-2025 
    litellm_params:
      model: command-a-03-2025
      api_key: "os.environ/COHERE_API_KEY"
```

```bash
litellm --config /path/to/config.yaml
```


### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell showLineNumbers
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer <your-litellm-api-key>' \
--data ' {
      "model": "command-a-03-2025",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python showLineNumbers
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy
response = client.chat.completions.create(model="command-a-03-2025", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)

```
</TabItem>
</Tabs>


## Supported Models
| Model Name | Function Call |
|------------|----------------|
| command-a-03-2025 | `litellm.completion('command-a-03-2025', messages)` |
| command-r-plus-08-2024 | `litellm.completion('command-r-plus-08-2024', messages)` |  
| command-r-08-2024 | `litellm.completion('command-r-08-2024', messages)` |
| command-r-plus | `litellm.completion('command-r-plus', messages)` |  
| command-r | `litellm.completion('command-r', messages)` |
| command-light | `litellm.completion('command-light', messages)` |  
| command-nightly | `litellm.completion('command-nightly', messages)` |


## Embedding

```python
from litellm import embedding
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = embedding(
    model="embed-english-v3.0", 
    input=["good morning from litellm", "this is another item"], 
)
```

### Setting - Input Type for v3 models
v3 Models have a required parameter: `input_type`. LiteLLM defaults to `search_document`. It can be one of the following four values:

- `input_type="search_document"`: (default) Use this for texts (documents) you want to store in your vector database
- `input_type="search_query"`: Use this for search queries to find the most relevant documents in your vector database
- `input_type="classification"`: Use this if you use the embeddings as an input for a classification system
- `input_type="clustering"`: Use this if you use the embeddings for text clustering

https://txt.cohere.com/introducing-embed-v3/


```python
from litellm import embedding
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = embedding(
    model="embed-english-v3.0", 
    input=["good morning from litellm", "this is another item"], 
    input_type="search_document" 
)
```

### Supported Embedding Models
| Model Name               | Function Call                                                |
|--------------------------|--------------------------------------------------------------|
| embed-english-v3.0       | `embedding(model="embed-english-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-light-v3.0 | `embedding(model="embed-english-light-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-v3.0  | `embedding(model="embed-multilingual-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-light-v3.0 | `embedding(model="embed-multilingual-light-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-v2.0       | `embedding(model="embed-english-v2.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-light-v2.0 | `embedding(model="embed-english-light-v2.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-v2.0  | `embedding(model="embed-multilingual-v2.0", input=["good morning from litellm", "this is another item"])` |

## Rerank 

### Usage

LiteLLM supports the v1 and v2 clients for Cohere rerank. By default, the `rerank` endpoint uses the v2 client, but you can specify the v1 client by explicitly calling `v1/rerank`

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK Usage">

```python
from litellm import rerank
import os

os.environ["COHERE_API_KEY"] = "sk-.."

query = "What is the capital of the United States?"
documents = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Washington, D.C. is the capital of the United States.",
    "Capital punishment has existed in the United States since before it was a country.",
]

response = rerank(
    model="cohere/rerank-english-v3.0",
    query=query,
    documents=documents,
    top_n=3,
)
print(response)
```
</TabItem>

<TabItem value="proxy" label="LiteLLM Proxy Usage">

LiteLLM provides an cohere api compatible `/rerank` endpoint for Rerank calls.

**Setup**

Add this to your litellm proxy config.yaml

```yaml
model_list:
  - model_name: Salesforce/Llama-Rank-V1
    litellm_params:
      model: together_ai/Salesforce/Llama-Rank-V1
      api_key: os.environ/TOGETHERAI_API_KEY
  - model_name: rerank-english-v3.0
    litellm_params:
      model: cohere/rerank-english-v3.0
      api_key: os.environ/COHERE_API_KEY
```

Start litellm

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

Test request

```bash
curl http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "documents": [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Washington, D.C. is the capital of the United States.",
        "Capital punishment has existed in the United States since before it was a country."
    ],
    "top_n": 3
  }'
```

</TabItem>
</Tabs>