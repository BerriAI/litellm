
import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure OpenAI

## Overview

| Property | Details |
|-------|-------|
| Description | Azure OpenAI Service provides REST API access to OpenAI's powerful language models including o1, o1-mini, GPT-5, GPT-4o, GPT-4o mini, GPT-4 Turbo with Vision, GPT-4, GPT-3.5-Turbo, and Embeddings model series. Also supports Claude models via Azure Foundry. |
| Provider Route on LiteLLM | `azure/`, [`azure/o_series/`](#o-series-models), [`azure/gpt5_series/`](#gpt-5-models), [`azure/claude-*`](./azure_anthropic) (Claude models via Azure Foundry) |
| Supported Operations | [`/chat/completions`](#azure-openai-chat-completion-models), [`/responses`](./azure_responses), [`/completions`](#azure-instruct-models), [`/embeddings`](./azure_embedding), [`/audio/speech`](azure_speech), [`/audio/transcriptions`](../audio_transcription), `/fine_tuning`, [`/batches`](#azure-batches-api), `/files`, [`/images`](../image_generation#azure-openai-image-generation-models), [`/anthropic/v1/messages`](./azure_anthropic) |
| Link to Provider Doc | [Azure OpenAI ↗](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview), [Azure Foundry Claude ↗](https://learn.microsoft.com/en-us/azure/ai-services/foundry-models/claude)

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

:::info Azure Foundry Claude Models

Azure also supports Claude models via Azure Foundry. Use `azure/claude-*` model names (e.g., `azure/claude-sonnet-4-5`) with Azure authentication. See the [Azure Anthropic documentation](./azure_anthropic) for details.

:::

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


### Setting API Version

You can set the `api_version` for Azure OpenAI in your proxy config.yaml in the following ways

#### Option 1: Per Model Configuration

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/my-gpt4-deployment
      api_base: https://your-resource.openai.azure.com/
      api_version: "2024-08-01-preview"  # Set version per model
      api_key: os.environ/AZURE_API_KEY
```





## Azure OpenAI Chat Completion Models

:::tip

**We support ALL Azure models, just set `model=azure/<your deployment name>` as a prefix when sending litellm requests**

:::

| Model Name       | Function Call                          |
|------------------|----------------------------------------|
| o1-mini | `response = completion(model="azure/<your deployment name>", messages=messages)` |
| o1-preview | `response = completion(model="azure/<your deployment name>", messages=messages)` |
| gpt-5 | `response = completion(model="azure/<your deployment name>", messages=messages)` |
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
                                "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
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

## O-Series Models

Azure OpenAI O-Series models are supported on LiteLLM. 

LiteLLM routes any deployment name with `o1` or `o3` in the model name, to the O-Series [transformation](https://github.com/BerriAI/litellm/blob/91ed05df2962b8eee8492374b048d27cc144d08c/litellm/llms/azure/chat/o1_transformation.py#L4) logic.

To set this explicitly, set `model` to `azure/o_series/<your-deployment-name>`.

**Automatic Routing**

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

litellm.completion(model="azure/my-o3-deployment", messages=[{"role": "user", "content": "Hello, world!"}]) # 👈 Note: 'o3' in the deployment name
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: o3-mini
    litellm_params:
      model: azure/o3-model
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
```

</TabItem>
</Tabs>

**Explicit Routing**

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

litellm.completion(model="azure/o_series/my-random-deployment-name", messages=[{"role": "user", "content": "Hello, world!"}]) # 👈 Note: 'o_series/' in the deployment name
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: o3-mini
    litellm_params:
      model: azure/o_series/my-random-deployment-name
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
```
</TabItem>
</Tabs>


## GPT-5 Models

| Property | Details |
|-------|-------|
| Description | Azure OpenAI GPT-5 models |
| Provider Route on LiteLLM | `azure/gpt5_series/<custom-name>` or `azure/gpt-5-deployment-name` |

LiteLLM supports using Azure GPT-5 models in one of the two ways:
1. Explicit Routing: `model = azure/gpt5_series/<deployment-name>`. In this scenario the model onboarded to litellm follows the format `model=azure/gpt5_series/<deployment-name>`.
2. Inferred Routing (If the azure deployment name contains `gpt-5` in the name): `model = azure/gpt-5-mini`. In this scenario the model onboarded to litellm follows the format `model=azure/gpt-5-mini`.

#### Explicit Routing
Use `azure/gpt5_series/<deployment-name>` for explicit GPT-5 model routing. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

response = litellm.completion(
    model="azure/gpt5_series/my-gpt-5-deployment",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: gpt-5
    litellm_params:
      model: azure/gpt5_series/my-gpt-5-deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
```

</TabItem>
</Tabs>

#### Inferred Routing (gpt-5 in the deployment name)
If your Azure deployment name contains `gpt-5`, LiteLLM automatically recognizes it as a GPT-5 model.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

# Deployment name contains 'gpt-5' - automatically inferred
response = litellm.completion(
    model="azure/my-gpt-5-deployment", 
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: gpt-5-mini
    litellm_params:
      model: azure/my-gpt-5-deployment  # deployment name contains 'gpt-5'
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
```

</TabItem>
</Tabs>






## Azure Audio Model

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

response = completion(
    model="azure/azure-openai-4o-audio",
    messages=[
      {
        "role": "user",
        "content": "I want to try out speech to speech"
      }
    ],
    modalities=["text","audio"],
    audio={"voice": "alloy", "format": "wav"}
)

print(response)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: azure-openai-4o-audio
    litellm_params:
      model: azure/azure-openai-4o-audio
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: os.environ/AZURE_API_VERSION
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!


```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "azure-openai-4o-audio",
    "messages": [{"role": "user", "content": "I want to try out speech to speech"}],
    "modalities": ["text","audio"],
    "audio": {"voice": "alloy", "format": "wav"}
  }'
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

## **Authentication**


### Entra ID - use `azure_ad_token`

This is a walkthrough on how to use Azure Active Directory Tokens - Microsoft Entra ID to make `litellm.completion()` calls.  
> **Note:** You can follow the same process below to use Azure Active Directory Tokens for all other Azure endpoints (e.g., chat, embeddings, image, audio, etc.) with LiteLLM.

Step 1 - Download Azure CLI 
Installation instructions: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
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

### Entra ID - use tenant_id, client_id, client_secret

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
      azure_scope: os.environ/AZURE_SCOPE  # defaults to "https://cognitiveservices.azure.com/.default"
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

### Entra ID - use client_id, username, password

Here is an example of setting up `client_id`, `azure_username`, `azure_password` in your litellm proxy `config.yaml`
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      client_id: os.environ/AZURE_CLIENT_ID
      azure_username: os.environ/AZURE_USERNAME
      azure_password: os.environ/AZURE_PASSWORD
      azure_scope: os.environ/AZURE_SCOPE  # defaults to "https://cognitiveservices.azure.com/.default"
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


### Azure AD Token Refresh - `DefaultAzureCredential`

Use this if you want to use Azure `DefaultAzureCredential` for Authentication on your requests. `DefaultAzureCredential` automatically discovers and uses available Azure credentials from multiple sources.

<Tabs>
<TabItem value="sdk" label="SDK">

**Option 1: Explicit DefaultAzureCredential (Recommended)**
```python
from litellm import completion
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# DefaultAzureCredential automatically discovers credentials from:
# - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
# - Managed Identity (AKS, Azure VMs, etc.)
# - Azure CLI credentials
# - And other Azure identity sources
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

response = completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    azure_ad_token_provider=token_provider,
    messages = [{"role": "user", "content": "good morning"}],
)
```

**Option 2: LiteLLM Auto-Fallback to DefaultAzureCredential**
```python
import litellm

# Enable automatic fallback to DefaultAzureCredential
litellm.enable_azure_ad_token_refresh = True

response = litellm.completion(
    model = "azure/<your deployment name>",
    api_base = "",
    api_version = "",
    messages = [{"role": "user", "content": "good morning"}],
)
```

</TabItem>
<TabItem value="proxy" label="PROXY config.yaml">

**Scenario 1: With Environment Variables (Traditional)**

1. Add relevant env vars

```bash
export AZURE_TENANT_ID=""
export AZURE_CLIENT_ID=""
export AZURE_CLIENT_SECRET=""
```

2. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/your-deployment-name
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/

litellm_settings:
    enable_azure_ad_token_refresh: true # 👈 KEY CHANGE
```

**Scenario 2: Managed Identity (AKS, Azure VMs) - No Hard-coded Credentials Required**

Perfect for AKS clusters, Azure VMs, or other managed environments where Azure automatically injects credentials.

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/your-deployment-name
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/

litellm_settings:
    enable_azure_ad_token_refresh: true # 👈 KEY CHANGE
```

**Scenario 3: Azure CLI Authentication**

If you're authenticated via `az login`, no additional configuration needed:

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/your-deployment-name
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/

litellm_settings:
    enable_azure_ad_token_refresh: true # 👈 KEY CHANGE
```

3. Start proxy

```bash
litellm --config /path/to/config.yaml
```

**How it works**: 
- LiteLLM first tries Service Principal authentication (if environment variables are available)
- If that fails, it automatically falls back to `DefaultAzureCredential`
- `DefaultAzureCredential` will use Managed Identity, Azure CLI credentials, or other available Azure identity sources
- This eliminates the need for hard-coded credentials in managed environments like AKS

</TabItem>
</Tabs>


## **Azure Batches API**

| Property | Details |
|-------|-------|
| Description | Azure OpenAI Batches API |
| `custom_llm_provider` on LiteLLM | `azure/` |
| Supported Operations | `/v1/batches`, `/v1/files` |
| Azure OpenAI Batches API | [Azure OpenAI Batches API ↗](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch) |
| Cost Tracking, Logging Support | ✅ LiteLLM will log, track cost for Batch API Requests |


### Quick Start

Just add the azure env vars to your environment. 

```bash
export AZURE_API_KEY=""
export AZURE_API_BASE=""
```

<Tabs>
<TabItem value="proxy" label="LiteLLM PROXY Server">

**1. Upload a File**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
from openai import OpenAI

# Initialize the client
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-api-key"
)

batch_input_file = client.files.create(
    file=open("mydata.jsonl", "rb"),
    purpose="batch",
    extra_headers={"custom-llm-provider": "azure"}
)
file_id = batch_input_file.id
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@mydata.jsonl"
```

</TabItem>
</Tabs>

**Example File Format**
```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
{"custom_id": "task-2", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "What is Altair Basic?"}]}}
```

**2. Create a Batch Request**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
batch = client.batches.create( # re use client from above
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "My batch job"},
    extra_headers={"custom-llm-provider": "azure"}
)
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/batches \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```
</TabItem>
</Tabs>

**3. Retrieve a Batch**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
retrieved_batch = client.batches.retrieve(
    batch.id,
    extra_headers={"custom-llm-provider": "azure"}
)
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/batches/batch_abc123 \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
```

</TabItem>
</Tabs>

**4. Cancel a Batch**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
cancelled_batch = client.batches.cancel(
    batch.id,
    extra_headers={"custom-llm-provider": "azure"}
)
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/batches/batch_abc123/cancel \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST
```

</TabItem>
</Tabs>

**5. List Batches**

<Tabs>
<TabItem value="sdk" label="OpenAI Python SDK">

```python
client.batches.list(extra_headers={"custom-llm-provider": "azure"})
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/batches?limit=2 \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json"
```
</TabItem>
</Tabs>
</TabItem>
<TabItem value="sdk" label="LiteLLM SDK">

**1. Create File for Batch Completion**

```python
from litellm
import os 

os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""

file_name = "azure_batch_completions.jsonl"
_current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_current_dir, file_name)
file_obj = await litellm.acreate_file(
    file=open(file_path, "rb"),
    purpose="batch",
    custom_llm_provider="azure",
)
print("Response from creating file=", file_obj)
```

**2. Create Batch Request**

```python
create_batch_response = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=batch_input_file_id,
    custom_llm_provider="azure",
    metadata={"key1": "value1", "key2": "value2"},
)

print("response from litellm.create_batch=", create_batch_response)
```

**3. Retrieve Batch and File Content**

```python
retrieved_batch = await litellm.aretrieve_batch(
    batch_id=create_batch_response.id, 
    custom_llm_provider="azure"
)
print("retrieved batch=", retrieved_batch)

# Get file content
file_content = await litellm.afile_content(
    file_id=batch_input_file_id, 
    custom_llm_provider="azure"
)
print("file content = ", file_content)
```

**4. List Batches**

```python
list_batches_response = litellm.list_batches(
    custom_llm_provider="azure", 
    limit=2
)
print("list_batches_response=", list_batches_response)
```

</TabItem>
</Tabs>

### [Health Check Azure Batch models](./proxy/health.md#batch-models-azure-only)


### [BETA] Loadbalance Multiple Azure Deployments 
In your config.yaml, set `enable_loadbalancing_on_batch_endpoints: true`

```yaml
model_list:
  - model_name: "batch-gpt-4o-mini"
    litellm_params:
      model: "azure/gpt-4o-mini"
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
    model_info:
      mode: batch

litellm_settings:
  enable_loadbalancing_on_batch_endpoints: true # 👈 KEY CHANGE
```

Note: This works on `{PROXY_BASE_URL}/v1/files` and `{PROXY_BASE_URL}/v1/batches`.
Note: Response is in the OpenAI-format. 

1. Upload a file 

Just set `model: batch-gpt-4o-mini` in your .jsonl.

```bash
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@mydata.jsonl"
```

**Example File**

Note: `model` should be your azure deployment name.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "batch-gpt-4o-mini", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "batch-gpt-4o-mini", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
{"custom_id": "task-2", "method": "POST", "url": "/chat/completions", "body": {"model": "batch-gpt-4o-mini", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "What is Altair Basic?"}]}}
```

Expected Response (OpenAI-compatible)

```bash
{"id":"file-f0be81f654454113a922da60acb0eea6",...}
```

2. Create a batch 

```bash
curl http://0.0.0.0:4000/v1/batches \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-f0be81f654454113a922da60acb0eea6",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h",
    "model: "batch-gpt-4o-mini"
  }'
```

Expected Response: 

```bash
{"id":"batch_94e43f0a-d805-477d-adf9-bbb9c50910ed",...}
```

3. Retrieve a batch 

```bash
curl http://0.0.0.0:4000/v1/batches/batch_94e43f0a-d805-477d-adf9-bbb9c50910ed \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
```


Expected Response: 

```
{"id":"batch_94e43f0a-d805-477d-adf9-bbb9c50910ed",...}
```

4. List batch

```bash
curl http://0.0.0.0:4000/v1/batches?limit=2 \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json"
```

Expected Response:

```bash
{"data":[{"id":"batch_R3V...}
```

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


### Tool Calling / Function Calling

See a detailed walthrough of parallel function calling with litellm [here](https://docs.litellm.ai/docs/completion/function_call)


<Tabs>
<TabItem value="sdk" label="SDK">

```python
# set Azure env variables
import os
import litellm
import json

os.environ['AZURE_API_KEY'] = "" # litellm reads AZURE_API_KEY from .env and sends the request
os.environ['AZURE_API_BASE'] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ['AZURE_API_VERSION'] = "2023-07-01-preview"

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
    messages=[{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}],
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("\nLLM Response1:\n", response)
response_message = response.choices[0].message
tool_calls = response.choices[0].message.tool_calls
print("\nTool Choice:\n", tool_calls)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: azure-gpt-3.5
    litellm_params:
      model: azure/chatgpt-functioncalling
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
```

2. Start proxy

```bash
litellm --config config.yaml
```

3. Test it

```bash
curl -L -X POST 'http://localhost:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "azure-gpt-3.5",
    "messages": [
        {
            "role": "user",
            "content": "Hey, how'\''s it going? Thinking long and hard before replying - what is the meaning of the world and life itself"
        }
    ]
}'
```




</TabItem>
</Tabs>
### Spend Tracking for Azure OpenAI Models (PROXY)

Set base model for cost tracking azure image-gen call

#### Image Generation 

```yaml
model_list: 
  - model_name: dall-e-3
    litellm_params:
        model: azure/dall-e-3-test
        api_version: 2023-06-01-preview
        api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
        api_key: os.environ/AZURE_API_KEY
        base_model: dall-e-3 # 👈 set dall-e-3 as base model
    model_info:
        mode: image_generation
```

#### Chat Completions / Embeddings

**Problem**: Azure returns `gpt-4` in the response when `azure/gpt-4-1106-preview` is used. This leads to inaccurate cost tracking

**Solution** ✅ :  Set `base_model` on your config so litellm uses the correct model for calculating azure cost

Get the base model name from [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)

Example config with `base_model`
```yaml
model_list:
  - model_name: azure-gpt-3.5
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
    model_info:
      base_model: azure/gpt-4-1106-preview
```
