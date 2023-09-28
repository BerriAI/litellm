import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Proxy Server

Use this to spin up a proxy api to translate openai api calls to any non-openai model (e.g. Huggingface, TogetherAI, Ollama, etc.)

This works for async + streaming as well. 

Works with **ALL MODELS** supported by LiteLLM. To see supported providers check out this list - [Provider List](https://docs.litellm.ai/docs/providers).

**Requirements** Make sure relevant keys are set in the local .env. 
## quick start
Call Huggingface models through your OpenAI proxy.

Run this in your CLI.
```shell 
$ pip install litellm
```
```shell
$ export HUGGINGFACE_API_KEY=your-api-key # [OPTIONAL]

$ litellm --model huggingface/stabilityai/stablecode-instruct-alpha-3b
```

This will host a local proxy api at: **http://0.0.0.0:8000**

Other supported models:
<Tabs>
<TabItem value="anthropic" label="Anthropic">

```shell
$ export ANTHROPIC_API_KEY=my-api-key
$ litellm --model claude-instant-1
```

</TabItem>

<TabItem value="together_ai" label="TogetherAI">

```shell
$ export TOGETHERAI_API_KEY=my-api-key
$ litellm --model together_ai/lmsys/vicuna-13b-v1.5-16k
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```shell
$ export REPLICATE_API_KEY=my-api-key
$ litellm \
  --model replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3
```

</TabItem>

<TabItem value="petals" label="Petals">

```shell
$ litellm --model petals/meta-llama/Llama-2-70b-chat-hf
```

</TabItem>

<TabItem value="palm" label="Palm">

```shell
$ export PALM_API_KEY=my-palm-key
$ litellm --model palm/chat-bison
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```shell
$ export AZURE_API_KEY=my-api-key
$ export AZURE_API_BASE=my-api-base
$ export AZURE_API_VERSION=my-api-version

$ litellm --model azure/my-deployment-id
```

</TabItem>

<TabItem value="ai21" label="AI21">

```shell
$ export AI21_API_KEY=my-api-key
$ litellm --model j2-light
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```shell
$ export COHERE_API_KEY=my-api-key
$ litellm --model command-nightly
```

</TabItem>

</Tabs>

[**Jump to Code**](https://github.com/BerriAI/litellm/blob/fef4146396d5d87006259e00095a62e3900d6bb4/litellm/proxy.py#L36)

## setting api base
For **local model** or model on **custom endpoint**

Pass in the api_base as well

```shell
litellm --model huggingface/meta-llama/llama2 --api_base https://my-endpoint.huggingface.cloud
```

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your openai key

response = completion(
  model="gpt-3.5-turbo", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```python
from litellm import completion
import os

## set ENV variables
os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your openai key

response = completion(
  model="claude-2", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="vertex" label="VertexAI">

```python
from litellm import completion
import os

# auth: run 'gcloud auth application-default'
os.environ["VERTEX_PROJECT"] = "hardy-device-386718"
os.environ["VERTEX_LOCATION"] = "us-central1"

response = completion(
  model="chat-bison", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="hugging" label="HuggingFace">

```python
from litellm import completion 
import os

os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key" 

# e.g. Call 'WizardLM/WizardCoder-Python-34B-V1.0' hosted on HF Inference endpoints
response = completion(
  model="huggingface/WizardLM/WizardCoder-Python-34B-V1.0",
  messages=[{ "content": "Hello, how are you?","role": "user"}], 
  api_base="https://my-endpoint.huggingface.cloud"
)

print(response)
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```python
from litellm import completion
import os

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
response = completion(
  "azure/<your_deployment_id>", 
  messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>


<TabItem value="ollama" label="Ollama">

```python
from litellm import completion

response = completion(
            model="ollama/llama2", 
            messages = [{ "content": "Hello, how are you?","role": "user"}], 
            api_base="http://localhost:11434"
)
```
</TabItem>

</Tabs>

## test it 

```curl 
curl --location 'http://0.0.0.0:8000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "messages": [
    {
      "role": "user", 
      "content": "what do you know?"
    }
  ], 
}'
```

## tutorial - using with aider 
[Aider](https://github.com/paul-gauthier/aider) is an AI pair programming in your terminal.

But it only accepts OpenAI API Calls. 

In this tutorial we'll use Aider with WizardCoder (hosted on HF Inference Endpoints).

[NOTE]: To learn how to deploy a model on Huggingface 

### Step 1: Install aider and litellm
```shell 
$ pip install aider-chat litellm
```

### Step 2: Spin up local proxy
Save your huggingface api key in your local environment (can also do this via .env)

```shell
$ export HUGGINGFACE_API_KEY=my-huggingface-api-key
```

Point your local proxy to your model endpoint

```shell 
$ litellm \
  --model huggingface/WizardLM/WizardCoder-Python-34B-V1.0 \
  --api_base https://my-endpoint.huggingface.com
```
This will host a local proxy api at: **http://0.0.0.0:8000**

### Step 3: Replace openai api base in Aider
Aider lets you set the openai api base. So lets point it to our proxy instead. 

```shell
$ aider --openai-api-base http://0.0.0.0:8000
```



