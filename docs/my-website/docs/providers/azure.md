
import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure OpenAI
## API Keys, Params
api_key, api_base, api_version etc can be passed directly to `litellm.completion` - see here or set as `litellm.api_key` params see here
```python
import os
os.environ["AZURE_API_KEY"] = "" # "my-azure-api-key"
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2023-05-15"

# optional
os.environ["AZURE_AD_TOKEN"] = ""
os.environ["AZURE_API_TYPE"] = ""
```

## **Usage - LiteLLM Python SDK**
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_Azure_OpenAI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Completion - using .env variables

```python
from litellm import completion

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
response = completion(
    model = "azure/<your_deployment_name>", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

### Completion - using api_key, api_base, api_version

```python
import litellm

# azure call
response = litellm.completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    api_key = "",                                       # azure api key
    messages = [{"role": "user", "content": "good morning"}],
)
```

### Completion - using azure_ad_token, api_base, api_version

```python
import litellm

# azure call
response = litellm.completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    azure_ad_token="", 									# azure_ad_token 
    messages = [{"role": "user", "content": "good morning"}],
)
```


## **Usage - LiteLLM Proxy Server**

Here's how to call Azure OpenAI models with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export AZURE_API_KEY=""
```

### 2. Start the proxy 

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      api_key: os.environ/AZURE_API_KEY # The `os.environ/` prefix tells litellm to read this from the env.
```

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
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

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
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
    model = "gpt-3.5-turbo",
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



## Azure OpenAI Chat Completion Models

:::tip

**We support ALL Azure models, just set `model=azure/<your deployment name>` as a prefix when sending litellm requests**

:::

| Model Name       | Function Call                          |
|------------------|----------------------------------------|
| o1-mini | `response = completion(model="azure/<your deployment name>", messages=messages)` |
| o1-preview | `response = completion(model="azure/<your deployment name>", messages=messages)` |
| gpt-4o-mini            | `completion('azure/<your deployment name>', messages)`         |
| gpt-4o            | `completion('azure/<your deployment name>', messages)`         |
| gpt-4            | `completion('azure/<your deployment name>', messages)`         |
| gpt-4-0314            | `completion('azure/<your deployment name>', messages)`         | 
| gpt-4-0613            | `completion('azure/<your deployment name>', messages)`         |
| gpt-4-32k            | `completion('azure/<your deployment name>', messages)`         | 
| gpt-4-32k-0314            | `completion('azure/<your deployment name>', messages)`         |
| gpt-4-32k-0613            | `completion('azure/<your deployment name>', messages)`         | 
| gpt-4-1106-preview            | `completion('azure/<your deployment name>', messages)`         | 
| gpt-4-0125-preview            | `completion('azure/<your deployment name>', messages)`         | 
| gpt-3.5-turbo    | `completion('azure/<your deployment name>', messages)` |
| gpt-3.5-turbo-0301    | `completion('azure/<your deployment name>', messages)` |
| gpt-3.5-turbo-0613    | `completion('azure/<your deployment name>', messages)` |
| gpt-3.5-turbo-16k    | `completion('azure/<your deployment name>', messages)` |
| gpt-3.5-turbo-16k-0613    | `completion('azure/<your deployment name>', messages)`

## Azure OpenAI Vision Models 
| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| gpt-4-vision   | `completion(model="azure/<your deployment name>", messages=messages)` |
| gpt-4o            | `completion('azure/<your deployment name>', messages)`         |

#### Usage
```python
import os 
from litellm import completion

os.environ["AZURE_API_KEY"] = "your-api-key"

# azure call
response = completion(
    model = "azure/<your deployment name>", 
    messages=[
        {
            "role": "user",
            "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                }
                            }
                        ]
        }
    ],
)

```

#### Usage - with Azure Vision enhancements

Note: **Azure requires the `base_url` to be set with `/extensions`** 

Example 
```python
base_url=https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions
# base_url="{azure_endpoint}/openai/deployments/{azure_deployment}/extensions"
```

**Usage**
```python
import os 
from litellm import completion

os.environ["AZURE_API_KEY"] = "your-api-key"

# azure call
response = completion(
            model="azure/gpt-4-vision",
            timeout=5,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://avatars.githubusercontent.com/u/29436595?v=4"
                            },
                        },
                    ],
                }
            ],
            base_url="https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions",
            api_key=os.getenv("AZURE_VISION_API_KEY"),
            enhancements={"ocr": {"enabled": True}, "grounding": {"enabled": True}},
            dataSources=[
                {
                    "type": "AzureComputerVision",
                    "parameters": {
                        "endpoint": "https://gpt-4-vision-enhancement.cognitiveservices.azure.com/",
                        "key": os.environ["AZURE_VISION_ENHANCE_KEY"],
                    },
                }
            ],
)
```

## Azure O1 Models

| Model Name          | Function Call                                      |
|---------------------|----------------------------------------------------|
| o1-mini | `response = completion(model="azure/<your deployment name>", messages=messages)` |
| o1-preview | `response = completion(model="azure/<your deployment name>", messages=messages)` |

Set `litellm.enable_preview_features = True` to use Azure O1 Models with streaming support. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

litellm.enable_preview_features = True # 👈 KEY CHANGE

response = litellm.completion(
    model="azure/<your deployment name>",
    messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
    stream=True
)

for chunk in response:
    print(chunk)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

1. Setup config.yaml
```yaml
model_list:
  - model_name: o1-mini
    litellm_params:
      model: azure/o1-mini
      api_base: "os.environ/AZURE_API_BASE"
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "os.environ/AZURE_API_VERSION"

litellm_settings:
    enable_preview_features: true # 👈 KEY CHANGE
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it 

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(model="o1-mini", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
],
stream=True)

for chunk in response:
    print(chunk)
```
</TabItem>
</Tabs>

## Azure Instruct Models

Use `model="azure_text/<your-deployment>"`

| Model Name          | Function Call                                      |
|---------------------|----------------------------------------------------|
| gpt-3.5-turbo-instruct | `response = completion(model="azure_text/<your deployment name>", messages=messages)` |
| gpt-3.5-turbo-instruct-0914 | `response = completion(model="azure_text/<your deployment name>", messages=messages)` |


```python
import litellm

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

response = litellm.completion(
    model="azure_text/<your-deployment-name",
    messages=[{"role": "user", "content": "What is the weather like in Boston?"}]
)

print(response)
```

## Azure Text to Speech (tts)

**LiteLLM PROXY**

```yaml
 - model_name: azure/tts-1
    litellm_params:
      model: azure/tts-1
      api_base: "os.environ/AZURE_API_BASE_TTS"
      api_key: "os.environ/AZURE_API_KEY_TTS"
      api_version: "os.environ/AZURE_API_VERSION" 
```

**LiteLLM SDK**

```python 
from litellm import completion

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
        model="azure/<your-deployment-name",
        voice="alloy",
        input="the quick brown fox jumped over the lazy dogs",
    )
response.stream_to_file(speech_file_path)
```

## **Authentication**


### Entrata ID - use `azure_ad_token`

This is a walkthrough on how to use Azure Active Directory Tokens - Microsoft Entra ID to make `litellm.completion()` calls 

Step 1 - Download Azure CLI 
Installation instructons: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
```shell
brew update && brew install azure-cli
```
Step 2 - Sign in using `az`
```shell
az login --output table
```

Step 3 - Generate azure ad token
```shell
az account get-access-token --resource https://cognitiveservices.azure.com
```

In this step you should see an `accessToken` generated
```shell
{
  "accessToken": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IjlHbW55RlBraGMzaE91UjIybXZTdmduTG83WSIsImtpZCI6IjlHbW55RlBraGMzaE91UjIybXZTdmduTG83WSJ9",
  "expiresOn": "2023-11-14 15:50:46.000000",
  "expires_on": 1700005846,
  "subscription": "db38de1f-4bb3..",
  "tenant": "bdfd79b3-8401-47..",
  "tokenType": "Bearer"
}
```

Step 4 - Make litellm.completion call with Azure AD token

Set `azure_ad_token` = `accessToken` from step 3 or set `os.environ['AZURE_AD_TOKEN']`


<Tabs>
<TabItem value="sdk" label="SDK">


```python
response = litellm.completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    azure_ad_token="", 									# your accessToken from step 3 
    messages = [{"role": "user", "content": "good morning"}],
)

```

</TabItem>
<TabItem value="proxy" label="PROXY config.yaml">

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      azure_ad_token: os.environ/AZURE_AD_TOKEN
```

</TabItem>
</Tabs>

### Entrata ID - use tenant_id, client_id, client_secret

Here is an example of setting up `tenant_id`, `client_id`, `client_secret` in your litellm proxy `config.yaml`
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      tenant_id: os.environ/AZURE_TENANT_ID
      client_id: os.environ/AZURE_CLIENT_ID
      client_secret: os.environ/AZURE_CLIENT_SECRET
```

Test it 

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```

Example video of using `tenant_id`, `client_id`, `client_secret` with LiteLLM Proxy Server

<iframe width="840" height="500" src="https://www.loom.com/embed/70d3f219ee7f4e5d84778b7f17bba506?sid=04b8ff29-485f-4cb8-929e-6b392722f36d" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

### Azure AD Token Refresh - `DefaultAzureCredential`

Use this if you want to use Azure `DefaultAzureCredential` for Authentication on your requests

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")


response = completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    azure_ad_token_provider=token_provider
    messages = [{"role": "user", "content": "good morning"}],
)
```

</TabItem>
<TabItem value="proxy" label="PROXY config.yaml">

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/your-deployment-name
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/

litellm_settings:
    enable_azure_ad_token_refresh: true # 👈 KEY CHANGE
```

</TabItem>
</Tabs>


## Advanced
### Azure API Load-Balancing

Use this if you're trying to load-balance across multiple Azure/OpenAI deployments. 

`Router` prevents failed requests, by picking the deployment which is below rate-limit and has the least amount of tokens used. 

In production, [Router connects to a Redis Cache](#redis-queue) to track usage across multiple deployments.

#### Quick Start

```python
pip install litellm
```

```python
from litellm import Router

model_list = [{ # list of model deployments 
	"model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-v-2", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
	"tpm": 240000,
	"rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-functioncalling", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
	"tpm": 240000,
	"rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-3.5-turbo", 
		"api_key": os.getenv("OPENAI_API_KEY"),
	},
	"tpm": 1000000,
	"rpm": 9000
}]

router = Router(model_list=model_list)

# openai.chat.completions.create replacement
response = router.completion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}]

print(response)
```

#### Redis Queue 

```python
router = Router(model_list=model_list, 
                redis_host=os.getenv("REDIS_HOST"), 
                redis_password=os.getenv("REDIS_PASSWORD"), 
                redis_port=os.getenv("REDIS_PORT"))

print(response)
```


### Parallel Function calling
See a detailed walthrough of parallel function calling with litellm [here](https://docs.litellm.ai/docs/completion/function_call)
```python
# set Azure env variables
import os
os.environ['AZURE_API_KEY'] = "" # litellm reads AZURE_API_KEY from .env and sends the request
os.environ['AZURE_API_BASE'] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ['AZURE_API_VERSION'] = "2023-07-01-preview"

import litellm
import json
# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

## Step 1: send the conversation and available functions to the model
messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = litellm.completion(
    model="azure/chatgpt-functioncalling", # model = azure/<your-azure-deployment-name>
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("\nLLM Response1:\n", response)
response_message = response.choices[0].message
tool_calls = response.choices[0].message.tool_calls
print("\nTool Choice:\n", tool_calls)
```

