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

```python showLineNumbers title="Chat Completion"
import os
from litellm import completion

os.environ["WATSONX_URL"] = ""
os.environ["WATSONX_APIKEY"] = ""

response = completion(
  model="watsonx/meta-llama/llama-3-1-8b-instruct",
  messages=[{ "content": "what is your favorite colour?","role": "user"}],
  project_id="<my-project-id>"
)
```

## Usage - Streaming
```python showLineNumbers title="Streaming"
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

## Usage - Models in deployment spaces

Models deployed to a deployment space (e.g.: tuned models) can be called using the `deployment/<deployment_id>` format.

```python showLineNumbers title="Deployment Space"
import litellm

response = litellm.completion(
    model="watsonx/deployment/<deployment_id>",
    messages=[{"content": "Hello, how are you?", "role": "user"}],
    space_id="<deployment_space_id>"
)
```

## Usage - Embeddings

```python showLineNumbers title="Embeddings"
from litellm import embedding

response = embedding(
    model="watsonx/ibm/slate-30m-english-rtrvr",
    input=["What is the capital of France?"],
    project_id="<my-project-id>"
)
```

## LiteLLM Proxy Usage 

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
```

</TabItem>
<TabItem value="config" label="config.yaml">

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: llama-3-8b
    litellm_params:
      model: watsonx/meta-llama/llama-3-8b-instruct
      api_key: "os.environ/WATSONX_API_KEY"
```
</TabItem>
</Tabs>

### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
      "model": "llama-3-8b",
      "messages": [
        {
          "role": "user",
          "content": "what is your favorite colour?"
        }
      ]
    }'
```
</TabItem>
<TabItem value="openai" label="OpenAI SDK">

```python showLineNumbers
import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="llama-3-8b", 
    messages=[{"role": "user", "content": "what is your favorite colour?"}]
)
print(response)
```
</TabItem>
</Tabs>


## Supported Models

| Model Name                         | Command                                                                                  |
|------------------------------------|------------------------------------------------------------------------------------------|
| Llama 3.1 8B Instruct              | `completion(model="watsonx/meta-llama/llama-3-1-8b-instruct", messages=messages)`        |
| Llama 2 70B Chat                   | `completion(model="watsonx/meta-llama/llama-2-70b-chat", messages=messages)`             |
| Granite 13B Chat V2                | `completion(model="watsonx/ibm/granite-13b-chat-v2", messages=messages)`                 |
| Mixtral 8X7B Instruct              | `completion(model="watsonx/ibm-mistralai/mixtral-8x7b-instruct-v01-q", messages=messages)` |

For all available models, see [watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx).

## Supported Embedding Models

| Model Name | Function Call                                                          |
|------------|------------------------------------------------------------------------|
| Slate 30m  | `embedding(model="watsonx/ibm/slate-30m-english-rtrvr", input=input)`  |
| Slate 125m | `embedding(model="watsonx/ibm/slate-125m-english-rtrvr", input=input)` |

For all available embedding models, see [watsonx.ai embedding documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx).


## Advanced

### Using Zen API Key

You can use a Zen API key for long-term authentication instead of generating IAM tokens. Pass it either as an environment variable or as a parameter:

```python
import os
from litellm import completion

# Option 1: Set as environment variable
os.environ["WATSONX_ZENAPIKEY"] = "your-zen-api-key"

response = completion(
    model="watsonx/ibm/granite-13b-chat-v2",
    messages=[{"content": "What is your favorite color?", "role": "user"}],
    project_id="your-project-id"
)

# Option 2: Pass as parameter
response = completion(
    model="watsonx/ibm/granite-13b-chat-v2",
    messages=[{"content": "What is your favorite color?", "role": "user"}],
    zen_api_key="your-zen-api-key",
    project_id="your-project-id"
)
```

**Using with LiteLLM Proxy via OpenAI client:**

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",  # LiteLLM proxy key
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="watsonx/ibm/granite-3-3-8b-instruct",
    messages=[{"role": "user", "content": "What is your favorite color?"}],
    max_tokens=2048,
    extra_body={
        "project_id": "your-project-id",
        "zen_api_key": "your-zen-api-key"
    }
)
```

See [IBM documentation](https://www.ibm.com/docs/en/watsonx/w-and-w/2.2.0?topic=keys-generating-zenapikey-authorization-tokens) for more information on generating Zen API keys.


