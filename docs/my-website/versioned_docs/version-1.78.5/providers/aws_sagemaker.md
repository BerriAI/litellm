import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem'

# AWS Sagemaker
LiteLLM supports All Sagemaker Huggingface Jumpstart Models

:::tip

**We support ALL Sagemaker models, just set `model=sagemaker/<any-model-on-sagemaker>` as a prefix when sending litellm requests**

:::


### API KEYS
```python
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""
```

### Usage
```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="sagemaker/<your-endpoint-name>", 
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            temperature=0.2,
            max_tokens=80
        )
```

### Usage - Streaming
Sagemaker currently does not support streaming - LiteLLM fakes streaming by returning chunks of the response string

```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b", 
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            temperature=0.2,
            max_tokens=80,
            stream=True,
        )
for chunk in response:
    print(chunk)
```


## **LiteLLM Proxy Usage**

Here's how to call Sagemaker with the LiteLLM Proxy Server

### 1. Setup config.yaml

```yaml
model_list:
  - model_name: jumpstart-model
    litellm_params:
      model: sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614
      aws_access_key_id: os.environ/CUSTOM_AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/CUSTOM_AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/CUSTOM_AWS_REGION_NAME
```

All possible auth params: 

```
aws_access_key_id: Optional[str],
aws_secret_access_key: Optional[str],
aws_session_token: Optional[str],
aws_region_name: Optional[str],
aws_session_name: Optional[str],
aws_profile_name: Optional[str],
aws_role_name: Optional[str],
aws_web_identity_token: Optional[str],
```

### 2. Start the proxy 

```bash
litellm --config /path/to/config.yaml
```
### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "jumpstart-model",
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

response = client.chat.completions.create(model="jumpstart-model", messages = [
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
    model = "jumpstart-model",
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

## Set temperature, top p, etc.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
  model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  temperature=0.7,
  top_p=1
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

**Set on yaml**

```yaml
model_list:
  - model_name: jumpstart-model
    litellm_params:
      model: sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614
      temperature: <your-temp>
      top_p: <your-top-p>
```

**Set on request**

```python

import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="jumpstart-model", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
],
temperature=0.7,
top_p=1
)

print(response)

```

</TabItem>
</Tabs>

## **Allow setting temperature=0** for Sagemaker

By default when `temperature=0` is sent in requests to LiteLLM, LiteLLM rounds up to `temperature=0.1` since Sagemaker fails most requests when `temperature=0`

If you want to send `temperature=0` for your model here's how to set it up (Since Sagemaker can host any kind of model, some models allow zero temperature)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
  model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  temperature=0,
  aws_sagemaker_allow_zero_temp=True,
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

**Set `aws_sagemaker_allow_zero_temp` on yaml**

```yaml
model_list:
  - model_name: jumpstart-model
    litellm_params:
      model: sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614
      aws_sagemaker_allow_zero_temp: true
```

**Set `temperature=0` on request**

```python

import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="jumpstart-model", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
],
temperature=0,
)

print(response)

```

</TabItem>
</Tabs>

## Pass provider-specific params 

If you pass a non-openai param to litellm, we'll assume it's provider-specific and send it as a kwarg in the request body. [See more](../completion/input.md#provider-specific-params)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
  model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  top_k=1 # 👈 PROVIDER-SPECIFIC PARAM
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

**Set on yaml**

```yaml
model_list:
  - model_name: jumpstart-model
    litellm_params:
      model: sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614
      top_k: 1 # 👈 PROVIDER-SPECIFIC PARAM
```

**Set on request**

```python

import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="jumpstart-model", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
],
temperature=0.7,
extra_body={
    top_k=1 # 👈 PROVIDER-SPECIFIC PARAM
}
)

print(response)

```

</TabItem>
</Tabs>


### Passing Inference Component Name

If you have multiple models on an endpoint, you'll need to specify the individual model names, do this via `model_id`.  

```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="sagemaker/<your-endpoint-name>", 
            model_id="<your-model-name",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            temperature=0.2,
            max_tokens=80
        )
```

### Passing credentials as parameters - Completion()
Pass AWS credentials as parameters to litellm.completion
```python
import os 
from litellm import completion

response = completion(
            model="sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            aws_access_key_id="",
            aws_secret_access_key="",
            aws_region_name="",
)
```

### Applying Prompt Templates
To apply the correct prompt template for your sagemaker deployment, pass in it's hf model name as well. 

```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b", 
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            hf_model_name="meta-llama/Llama-2-7b",
        )
```

You can also pass in your own [custom prompt template](../completion/prompt_formatting.md#format-prompt-yourself)


## Sagemaker Messages API 

Use route `sagemaker_chat/*` to route to Sagemaker Messages API

```
model: sagemaker_chat/<your-endpoint-name>
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
import litellm
from litellm import completion

litellm.set_verbose = True # 👈 SEE RAW REQUEST

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="sagemaker_chat/<your-endpoint-name>", 
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            temperature=0.2,
            max_tokens=80
        )
```

</TabItem>
<TabItem value="proxy" label="PROXY">

#### 1. Setup config.yaml 

```yaml
model_list:
  - model_name: "sagemaker-model"
    litellm_params:
      model: "sagemaker_chat/jumpstart-dft-hf-textgeneration1-mp-20240815-185614"
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
```

#### 2. Start the proxy 

```bash
litellm --config /path/to/config.yaml
```
#### 3. Test it


```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "sagemaker-model",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```

[**👉 See OpenAI SDK/Langchain/Llamaindex/etc. examples**](../proxy/user_keys.md#chatcompletions)

</TabItem>
</Tabs>


## Completion Models 


:::tip

**We support ALL Sagemaker models, just set `model=sagemaker/<any-model-on-sagemaker>` as a prefix when sending litellm requests**

:::

Here's an example of using a sagemaker model with LiteLLM 

| Model Name                    | Function Call                                                                                       |
|-------------------------------|-------------------------------------------------------------------------------------------|
| Your Custom Huggingface Model               | `completion(model='sagemaker/<your-deployment-name>', messages=messages)`        | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`      
| Meta Llama 2 7B               | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b', messages=messages)`        | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 7B (Chat/Fine-tuned)  | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b-f', messages=messages)`      | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 13B              | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-13b', messages=messages)`       | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 13B (Chat/Fine-tuned) | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-13b-f', messages=messages)`     | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 70B              | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-70b', messages=messages)`       | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 70B (Chat/Fine-tuned) | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-70b-b-f', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |

## Embedding Models

LiteLLM supports all Sagemaker Jumpstart Huggingface Embedding models. Here's how to call it: 

```python
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = litellm.embedding(model="sagemaker/<your-deployment-name>", input=["good morning from litellm", "this is another item"])
print(f"response: {response}")
```


