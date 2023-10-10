import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Proxy Server

A local, fast, and lightweight OpenAI-compatible server to call 100+ LLM APIs. 

:::info
We want to learn how we can make the proxy better! Meet the [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
::: 

## Usage 
```shell
pip install litellm
```
```shell 
$ litellm --model ollama/codellama 

#INFO: Ollama running on http://0.0.0.0:8000
```

### Test
In a new shell, run: 
```shell
$ litellm --test
``` 

### Replace openai base

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

## [Tutorial]: Use with Aider/AutoGen/Continue-Dev

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
        "api_base": "http://localhost:8000",  #litellm compatible endpoint
        "api_type": "open_ai",
        "api_key": "NULL", # just a placeholder
    }
]

response = oai.Completion.create(config_list=config_list, prompt="Hi")
print(response) # works fine

llm_config={
    "config_list": config_list,
}

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy")
user_proxy.initiate_chat(assistant, message="Plot a chart of META and TESLA stock price change YTD.", config_list=config_list)
```

Credits [@victordibia](https://github.com/microsoft/autogen/issues/45#issuecomment-1749921972) for this tutorial.
</TabItem>
<TabItem value="langroid" label="Langroid">

```python
pip install langroid
```

```python
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIGPT

# create the (Pydantic-derived) config class: Allows setting params via MYLLM_XXX env vars
MyLLMConfig = OpenAIGPTConfig.create(prefix="myllm") 

# instantiate the class, with the model name and context length
my_llm_config = MyLLMConfig(
    chat_model="local/localhost:8000", # "local/[URL where LiteLLM proxy is listening]
    chat_context_length=2048,  # adjust based on model
)

# create llm and interact with it 
from langroid.language_models.base import LLMMessage, Role

llm = OpenAIGPT(my_llm_config)
messages = [
    LLMMessage(content="You are a helpful assistant",  role=Role.SYSTEM),
    LLMMessage(content="What is the capital of Ontario?",  role=Role.USER),
],
response = mdl.chat(messages, max_tokens=50)

# Create an Agent with this LLM, wrap it in a Task, and run it as an interactive chat app:
from langroid.agent.base import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task

agent_config = ChatAgentConfig(llm=my_llm_config, name="my-llm-agent")
agent = ChatAgent(agent_config)

task = Task(agent, name="my-llm-task")
task.run() 
```

Credits [@pchalasani](https://github.com/pchalasani) for this tutorial.
</TabItem>
</Tabs>

:::note
**Contribute** Using this server with a project? Contribute your tutorial [here!](https://github.com/BerriAI/litellm)

::: 

## Advanced
### Configure Model

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

### Create a proxy for multiple LLMs
```shell 
$ litellm

#INFO: litellm proxy running on http://0.0.0.0:8000
```

#### Send a request to your proxy
```python
import openai 

openai.api_key = "any-string-here"
openai.api_base = "http://0.0.0.0:8080" # your proxy url

# call gpt-3.5-turbo
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey"}])

print(response)

# call ollama/llama2
response = openai.ChatCompletion.create(model="ollama/llama2", messages=[{"role": "user", "content": "Hey"}])

print(response)
```

### Tracking costs
By default litellm proxy writes cost logs to litellm/proxy/cost.log

How can the proxy be better? Let us know [here](https://github.com/BerriAI/litellm/issues)
```shell
2023-10-09 14:48:18 - Model gpt-3.5-turbo-0613 Cost: 0.000047000
2023-10-09 14:48:18 - Model gpt-3.5-turbo-0613 Cost: 0.000128700
2023-10-09 14:48:20 - Model gpt-3.5-turbo-0613 Cost: 0.000523011
2023-10-09 14:48:28 - Model ollama/llama2 Cost: 0.00004700
2023-10-09 14:48:35 - Model ollama/code-llama Cost: 0.00004700
2023-10-09 14:48:49 - Model azure/chatgpt-2 Cost: 0.000223011
```

You can view costs on the cli using 
```shell
litellm --cost
```

### Ollama Logs
Ollama calls can sometimes fail (out-of-memory errors, etc.). 

To see your logs just call 

```shell
$ curl 'http://0.0.0.0:8000/ollama_logs'
```

This will return your logs from `~/.ollama/logs/server.log`. 

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
</TabItem>
</Tabs>

## Support/ talk with founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

