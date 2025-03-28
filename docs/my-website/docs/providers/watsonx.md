import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# IBM watsonx.ai

LiteLLM supports all IBM [watsonx.ai](https://watsonx.ai/) foundational models and embeddings.

## Environment Variables
```python
os.environ["WATSONX_URL"] = ""  # (required) Base URL of your WatsonX instance
# (required) either one of the following:
os.environ["WATSONX_APIKEY"] = "" # IBM cloud API key
os.environ["WATSONX_TOKEN"] = "" # IAM auth token
# optional - can also be passed as params to completion() or embedding()
os.environ["WATSONX_PROJECT_ID"] = "" # Project ID of your WatsonX instance
os.environ["WATSONX_DEPLOYMENT_SPACE_ID"] = "" # ID of your deployment space to use deployed models
os.environ["WATSONX_ZENAPIKEY"] = "" # Zen API key (use for long-term api token)
```

See [here](https://cloud.ibm.com/apidocs/watsonx-ai#api-authentication) for more information on how to get an access token to authenticate to watsonx.ai.

## Usage

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_IBM_Watsonx.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
import os
from litellm import completion

os.environ["WATSONX_URL"] = ""
os.environ["WATSONX_APIKEY"] = ""

## Call WATSONX `/text/chat` endpoint - supports function calling
response = completion(
  model="watsonx/meta-llama/llama-3-1-8b-instruct",
  messages=[{ "content": "what is your favorite colour?","role": "user"}],
  project_id="<my-project-id>" # or pass with os.environ["WATSONX_PROJECT_ID"]
)

## Call WATSONX `/text/generation` endpoint - not all models support /chat route. 
response = completion(
  model="watsonx/ibm/granite-13b-chat-v2",
  messages=[{ "content": "what is your favorite colour?","role": "user"}],
  project_id="<my-project-id>"
)
```

## Usage - Streaming
```python
import os
from litellm import completion

os.environ["WATSONX_URL"] = ""
os.environ["WATSONX_APIKEY"] = ""
os.environ["WATSONX_PROJECT_ID"] = ""

response = completion(
  model="watsonx/meta-llama/llama-3-1-8b-instruct",
  messages=[{ "content": "what is your favorite colour?","role": "user"}],
  stream=True
)
for chunk in response:
  print(chunk)
```

#### Example Streaming Output Chunk
```json
{
  "choices": [
    {
      "finish_reason": null,
      "index": 0,
      "delta": {
        "content": "I don't have a favorite color, but I do like the color blue. What's your favorite color?"
      }
    }
  ],
  "created": null,
  "model": "watsonx/ibm/granite-13b-chat-v2",
  "usage": {
    "prompt_tokens": null,
    "completion_tokens": null,
    "total_tokens": null
  }
}
```

## Usage - Models in deployment spaces

Models that have been deployed to a deployment space (e.g.: tuned models) can be called using the `deployment/<deployment_id>` format (where `<deployment_id>` is the ID of the deployed model in your deployment space). 

The ID of your deployment space must also be set in the environment variable `WATSONX_DEPLOYMENT_SPACE_ID` or passed to the function as `space_id=<deployment_space_id>`. 

```python
import litellm
response = litellm.completion(
    model="watsonx/deployment/<deployment_id>",
    messages=[{"content": "Hello, how are you?", "role": "user"}],
    space_id="<deployment_space_id>"
)
```

## Usage - Embeddings

LiteLLM also supports making requests to IBM watsonx.ai embedding models. The credential needed for this is the same as for completion.

```python
from litellm import embedding

response = embedding(
    model="watsonx/ibm/slate-30m-english-rtrvr",
    input=["What is the capital of France?"],
    project_id="<my-project-id>"
)
print(response)
# EmbeddingResponse(model='ibm/slate-30m-english-rtrvr', data=[{'object': 'embedding', 'index': 0, 'embedding': [-0.037463713, -0.02141933, -0.02851813, 0.015519324, ..., -0.0021367231, -0.01704561, -0.001425816, 0.0035238306]}], object='list', usage=Usage(prompt_tokens=8, total_tokens=8))
```

## OpenAI Proxy Usage 

Here's how to call IBM watsonx.ai with the LiteLLM Proxy Server

### 1. Save keys in your environment

```bash
export WATSONX_URL=""
export WATSONX_APIKEY=""
export WATSONX_PROJECT_ID=""
```

### 2. Start the proxy 

<Tabs>
<TabItem value="cli" label="CLI">

```bash
$ litellm --model watsonx/meta-llama/llama-3-8b-instruct

# Server running on http://0.0.0.0:4000
```

</TabItem>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: llama-3-8b
    litellm_params:
      # all params accepted by litellm.completion()
      model: watsonx/meta-llama/llama-3-8b-instruct
      api_key: "os.environ/WATSONX_API_KEY" # does os.getenv("WATSONX_API_KEY")
```
</TabItem>
</Tabs>

### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "llama-3-8b",
      "messages": [
        {
          "role": "user",
          "content": "what is your favorite colour?"
        }
      ]
    }
'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="llama-3-8b", messages=[
    {
        "role": "user",
        "content": "what is your favorite colour?"
    }
])

print(response)

```
</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000", # set openai_api_base to the LiteLLM Proxy
    model = "llama-3-8b",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```
</TabItem>
</Tabs>


## Authentication

### Passing credentials as parameters

You can also pass the credentials as parameters to the completion and embedding functions.

```python
import os
from litellm import completion

response = completion(
            model="watsonx/ibm/granite-13b-chat-v2",
            messages=[{ "content": "What is your favorite color?","role": "user"}],
            url="",
            api_key="",
            project_id=""
)
```


## Supported IBM watsonx.ai Models

Here are some examples of models available in IBM watsonx.ai that you can use with LiteLLM:

| Mode Name                          | Command                                                                                  |
|------------------------------------|------------------------------------------------------------------------------------------|
| Flan T5 XXL                        | `completion(model=watsonx/google/flan-t5-xxl, messages=messages)`                        |
| Flan Ul2                           | `completion(model=watsonx/google/flan-ul2, messages=messages)`                           |
| Mt0 XXL                            | `completion(model=watsonx/bigscience/mt0-xxl, messages=messages)`                        |
| Gpt Neox                           | `completion(model=watsonx/eleutherai/gpt-neox-20b, messages=messages)`                   |
| Mpt 7B Instruct2                   | `completion(model=watsonx/ibm/mpt-7b-instruct2, messages=messages)`                      |
| Starcoder                          | `completion(model=watsonx/bigcode/starcoder, messages=messages)`                         |
| Llama 2 70B Chat                   | `completion(model=watsonx/meta-llama/llama-2-70b-chat, messages=messages)`               |
| Llama 2 13B Chat                   | `completion(model=watsonx/meta-llama/llama-2-13b-chat, messages=messages)`               |
| Granite 13B Instruct               | `completion(model=watsonx/ibm/granite-13b-instruct-v1, messages=messages)`               |
| Granite 13B Chat                   | `completion(model=watsonx/ibm/granite-13b-chat-v1, messages=messages)`                   |
| Flan T5 XL                         | `completion(model=watsonx/google/flan-t5-xl, messages=messages)`                         |
| Granite 13B Chat V2                | `completion(model=watsonx/ibm/granite-13b-chat-v2, messages=messages)`                   |
| Granite 13B Instruct V2            | `completion(model=watsonx/ibm/granite-13b-instruct-v2, messages=messages)`               |
| Elyza Japanese Llama 2 7B Instruct | `completion(model=watsonx/elyza/elyza-japanese-llama-2-7b-instruct, messages=messages)`  |
| Mixtral 8X7B Instruct V01 Q        | `completion(model=watsonx/ibm-mistralai/mixtral-8x7b-instruct-v01-q, messages=messages)` |


For a list of all available models in watsonx.ai, see [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&locale=en&audience=wdp).


## Supported IBM watsonx.ai Embedding Models

| Model Name | Function Call                                                          |
|------------|------------------------------------------------------------------------|
| Slate 30m  | `embedding(model="watsonx/ibm/slate-30m-english-rtrvr", input=input)`  |
| Slate 125m | `embedding(model="watsonx/ibm/slate-125m-english-rtrvr", input=input)` |


For a list of all available embedding models in watsonx.ai, see [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx).