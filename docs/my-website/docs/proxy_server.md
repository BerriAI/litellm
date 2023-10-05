import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Proxy Server

CLI Tool to create a LLM Proxy Server to translate openai api calls to any non-openai model (e.g. Huggingface, TogetherAI, Ollama, etc.) 100+ models [Provider List](https://docs.litellm.ai/docs/providers).

## Quick start
Call Huggingface models through your OpenAI proxy.

### Start Proxy
```shell
$ pip install litellm
```
```shell 
$ litellm --model huggingface/bigcode/starcoder

#INFO:     Uvicorn running on http://0.0.0.0:8000
```

This will host a local proxy api at: **http://0.0.0.0:8000**

### Test Proxy
Make a test ChatCompletion Request to your proxy
<Tabs>
<TabItem value="litellm" label="litellm cli">

```shell
litellm --test http://0.0.0.0:8000
```

</TabItem>
<TabItem value="openai" label="OpenAI">

```python
import openai 

openai.api_base = "http://0.0.0.0:8000"

print(openai.ChatCompletion.create(model="test", messages=[{"role":"user", "content":"Hey!"}]))
```

</TabItem>

<TabItem value="curl" label="curl">

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
</TabItem>
</Tabs>

#### Other supported models:
<Tabs>
<TabItem value="anthropic" label="Anthropic">

```shell
$ export ANTHROPIC_API_KEY=my-api-key
$ litellm --model claude-instant-1
```

</TabItem>

<TabItem value="huggingface" label="Huggingface">

```shell
$ export HUGGINGFACE_API_KEY=my-api-key #[OPTIONAL]
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


### Deploy Proxy

<Tabs>
<TabItem value="self-hosted" label="Self-Hosted">

**Step 1: Clone the repo**
```shell
git clone https://github.com/BerriAI/liteLLM-proxy.git
```

**Step 2: Put your API keys in .env** 
Copy the .env.template and put in the relevant keys (e.g. OPENAI_API_KEY="sk-..")

**Step 3: Test your proxy**
Start your proxy server
```shell
cd litellm-proxy && python3 main.py 
```

Make your first call 
```python
import openai 

openai.api_key = "sk-litellm-master-key"
openai.api_base = "http://0.0.0.0:8080"

response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey"}])

print(response)
```
</TabItem>
<TabItem value="litellm-hosted" label="LiteLLM-Hosted">

Deploy the proxy to https://api.litellm.ai

```shell 
$ export ANTHROPIC_API_KEY=sk-ant-api03-1..
$ litellm --model claude-instant-1 --deploy

#INFO:     Uvicorn running on https://api.litellm.ai/44508ad4
```

This will host a ChatCompletions API at: https://api.litellm.ai/44508ad4
#### Other supported models:
<Tabs>
<TabItem value="anthropic" label="Anthropic">

```shell
$ export ANTHROPIC_API_KEY=my-api-key
$ litellm --model claude-instant-1 --deploy
```

</TabItem>

<TabItem value="together_ai" label="TogetherAI">

```shell
$ export TOGETHERAI_API_KEY=my-api-key
$ litellm --model together_ai/lmsys/vicuna-13b-v1.5-16k --deploy
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```shell
$ export REPLICATE_API_KEY=my-api-key
$ litellm \
  --model replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3
  --deploy
```

</TabItem>

<TabItem value="petals" label="Petals">

```shell
$ litellm --model petals/meta-llama/Llama-2-70b-chat-hf --deploy
```

</TabItem>

<TabItem value="palm" label="Palm">

```shell
$ export PALM_API_KEY=my-palm-key
$ litellm --model palm/chat-bison --deploy
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```shell
$ export AZURE_API_KEY=my-api-key
$ export AZURE_API_BASE=my-api-base
$ export AZURE_API_VERSION=my-api-version

$ litellm --model azure/my-deployment-id --deploy
```

</TabItem>

<TabItem value="ai21" label="AI21">

```shell
$ export AI21_API_KEY=my-api-key
$ litellm --model j2-light --deploy
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```shell
$ export COHERE_API_KEY=my-api-key
$ litellm --model command-nightly --deploy
```

</TabItem>

</Tabs>

### Test Deployed Proxy
Make a test ChatCompletion Request to your proxy
<Tabs>
<TabItem value="litellm" label="litellm cli">

```shell
litellm --test https://api.litellm.ai/44508ad4
```

</TabItem>
<TabItem value="openai" label="OpenAI">

```python
import openai 

openai.api_base = "https://api.litellm.ai/44508ad4"

print(openai.ChatCompletion.create(model="test", messages=[{"role":"user", "content":"Hey!"}]))
```

</TabItem>

<TabItem value="curl" label="curl">

```curl 
curl --location 'https://api.litellm.ai/44508ad4/chat/completions' \
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
</TabItem>
</Tabs>
</TabItem>
</Tabs>

## Setting api base, temperature, max tokens

```shell
litellm --model huggingface/bigcode/starcoder \
  --api_base https://my-endpoint.huggingface.cloud \
  --max_tokens 250 \
  --temperature 0.5
```

**Ollama example**

```shell
$ litellm --model ollama/llama2 --api_base http://localhost:11434
```

## Tutorial - using HuggingFace LLMs with aider 
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



And that's it! 