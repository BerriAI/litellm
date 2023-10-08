import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Proxy Server

A local, fast, and lightweight OpenAI-compatible server to call 100+ LLM APIs. 

## usage 
```shell
pip install litellm
```
```shell 
$ litellm --model ollama/codellama 

#INFO: Ollama running on http://0.0.0.0:8000
```

### test
In a new shell, run: 
```shell
$ litellm --test
``` 

### replace openai base

```python
import openai 

openai.api_base = "http://0.0.0.0:8000"

print(openai.ChatCompletion.create(model="test", messages=[{"role":"user", "content":"Hey!"}]))
```

#### Other supported models:
<Tabs>
<TabItem value="vllm-local" label="VLLM">
Assuming you're running vllm locally

```shell
$ litellm --model vllm/facebook/opt-125m
```
</TabItem>
<TabItem value="openai-proxy" label="OpenAI Compatible Server">

```shell
$ litellm --model openai/<model_name> --api_base <your-api-base>
```
</TabItem>
<TabItem value="huggingface" label="Huggingface">

```shell
$ export HUGGINGFACE_API_KEY=my-api-key #[OPTIONAL]
$ litellm --model claude-instant-1
```

</TabItem>
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

$ litellm --model azure/my-deployment-name
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

## [tutorial]: Use with Aider/AutoGen/Continue-Dev

Here's how to use the proxy to test codellama/mistral/etc. models for different github repos 

```shell
pip install litellm
```

```shell
$ ollama pull codellama # OUR Local CodeLlama  

$ litellm --model ollama/codellama --temperature 0.3 --max_tokens 2048
```

Implementation for different repos 
<Tabs>
<TabItem value="aider" label="Aider">

```shell
$ pip install aider 

$ aider --openai-api-base http://0.0.0.0:8000 --openai-api-key fake-key
```
</TabItem>
<TabItem value="continue-dev" label="ContinueDev">

Continue-Dev brings ChatGPT to VSCode. See how to [install it here](https://continue.dev/docs/quickstart).

In the [config.py](https://continue.dev/docs/reference/Models/openai) set this as your default model.
```python
  default=OpenAI(
      api_key="IGNORED",
      model="fake-model-name",
      context_length=2048,
      api_base="http://your_litellm_hostname:8000"
  ),
```

Credits [@vividfog](https://github.com/jmorganca/ollama/issues/305#issuecomment-1751848077) for this tutorial. 
</TabItem>
<TabItem value="autogen" label="AutoGen">

```python
pip install pyautogen
```

```python
from autogen import AssistantAgent, UserProxyAgent, oai
config_list=[
    {
        "model": "my-fake-model",
        "api_base": "http://localhost:8000/v1",  #litellm compatible endpoint
        "api_type": "open_ai",
        "api_key": "NULL", # just a placeholder
    }
]

response = oai.Completion.create(config_list=config_list, prompt="Hi")
print(response) # works fine

assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user_proxy")
user_proxy.initiate_chat(assistant, message="Plot a chart of META and TESLA stock price change YTD.", config_list=config_list)
# fails with the error: openai.error.AuthenticationError: No API key provided.
```

Credits [@victordibia](https://github.com/microsoft/autogen/issues/45#issuecomment-1749921972) for this tutorial.
</TabItem>
</Tabs>

:::note
**Contribute** Using this server with a project? Contribute your tutorial here!

::: 

## Configure Model

To save api keys and/or customize model prompt, run: 
```shell
$ litellm --config
```
This will open a .env file that will store these values locally.

To set api base, temperature, and max tokens, add it to your cli command
```shell
litellm --model ollama/llama2 \
  --api_base http://localhost:11434 \
  --max_tokens 250 \
  --temperature 0.5
```

## Deploy Proxy

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
</TabItem>
</Tabs>

