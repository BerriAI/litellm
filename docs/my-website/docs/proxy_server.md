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

## [Tutorial]: Use with Continue-Dev/Aider/AutoGen/Langroid/etc.

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
<TabItem value="continue-dev" label="ContinueDev">

Continue-Dev brings ChatGPT to VSCode. See how to [install it here](https://continue.dev/docs/quickstart).

In the [config.py](https://continue.dev/docs/reference/Models/openai) set this as your default model.
```python
  default=OpenAI(
      api_key="IGNORED",
      model="fake-model-name",
      context_length=2048, # customize if needed for your model
      api_base="http://localhost:8000" # your proxy server url
  ),
```

Credits [@vividfog](https://github.com/jmorganca/ollama/issues/305#issuecomment-1751848077) for this tutorial. 
</TabItem>
<TabItem value="aider" label="Aider">

```shell
$ pip install aider 

$ aider --openai-api-base http://0.0.0.0:8000 --openai-api-key fake-key
```
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

# configure the LLM
my_llm_config = OpenAIGPTConfig(
    #format: "local/[URL where LiteLLM proxy is listening]
    chat_model="local/localhost:8000", 
    chat_context_length=2048,  # adjust based on model
)

# create llm, one-off interaction
llm = OpenAIGPT(my_llm_config)
response = mdl.chat("What is the capital of China?", max_tokens=50)

# Create an Agent with this LLM, wrap it in a Task, and 
# run it as an interactive chat app:
from langroid.agent.base import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task

agent_config = ChatAgentConfig(llm=my_llm_config, name="my-llm-agent")
agent = ChatAgent(agent_config)

task = Task(agent, name="my-llm-task")
task.run() 
```

Credits [@pchalasani](https://github.com/pchalasani) and [Langroid](https://github.com/langroid/langroid) for this tutorial.
</TabItem>
<TabItem value="gpt-pilot" label="GPT-Pilot">
GPT-Pilot helps you build apps with AI Agents. [For more](https://github.com/Pythagora-io/gpt-pilot)

In your .env set the openai endpoint to your local server. 

```
OPENAI_ENDPOINT=http://0.0.0.0:8000
OPENAI_API_KEY=my-fake-key
```
</TabItem>
<TabItem value="guidance" label="guidance">
A guidance language for controlling large language models.
https://github.com/guidance-ai/guidance

**NOTE:** Guidance sends additional params like `stop_sequences` which can cause some models to fail if they don't support it. 

**Fix**: Start your proxy using the `--drop_params` flag

```shell
litellm --model ollama/codellama --temperature 0.3 --max_tokens 2048 --drop_params
```

```python
import guidance

# set api_base to your proxy
# set api_key to anything
gpt4 = guidance.llms.OpenAI("gpt-4", api_base="http://0.0.0.0:8000", api_key="anything")

experts = guidance('''
{{#system~}}
You are a helpful and terse assistant.
{{~/system}}

{{#user~}}
I want a response to the following question:
{{query}}
Name 3 world-class experts (past or present) who would be great at answering this?
Don't answer the question yet.
{{~/user}}

{{#assistant~}}
{{gen 'expert_names' temperature=0 max_tokens=300}}
{{~/assistant}}
''', llm=gpt4)

result = experts(query='How can I be more productive?')
print(result)
```
</TabItem>
</Tabs>

:::note
**Contribute** Using this server with a project? Contribute your tutorial [here!](https://github.com/BerriAI/litellm)

::: 

## Advanced

### Multiple LLMs
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

### Logs

```shell
$ litellm --logs
```

This will return the most recent log (the call that went to the LLM API + the received response).

All logs are saved to a file called `api_logs.json` in the current directory. 

### Deploy Proxy

<Tabs>
<TabItem value="self-hosted" label="Self-Hosted">

**Step 1: Clone the repo**
```shell
git clone https://github.com/BerriAI/litellm.git
```

**Step 2: Modify `secrets_template.toml`**  
Add your api keys / configure default model: 
```shell
[keys]
OPENAI_API_KEY="sk-..."

[general]
default_model = "gpt-3.5-turbo"
```

**Step 3: Deploy Proxy**  
```shell
docker build -t litellm . && docker run -p 8000:8000 litellm
```
</TabItem>
<TabItem value="docker" label="Ollama/OpenAI Docker">
Use this to deploy local models with Ollama that's OpenAI-compatible. 

It works for models like Mistral, Llama2, CodeLlama, etc. (any model supported by [Ollama](https://ollama.ai/library))

**usage**
```shell
docker run --name ollama litellm/ollama
```

More details 👉 https://hub.docker.com/r/litellm/ollama
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

### Configure Proxy

If you need to: 
* save API keys 
* set litellm params (e.g. drop unmapped params, set fallback models, etc.)
* set model-specific params (max tokens, temperature, api base, prompt template)

You can do set these just for that session (via cli), or persist these across restarts (via config file).

#### Save API Keys 
```shell 
$ litellm --api_key OPENAI_API_KEY=sk-...
```
LiteLLM will save this to a locally stored config file, and persist this across sessions. 

LiteLLM Proxy supports all litellm supported api keys. To add keys for a specific provider, check this list:

<Tabs>
<TabItem value="huggingface" label="Huggingface">

```shell
$ litellm --add_key HUGGINGFACE_API_KEY=my-api-key #[OPTIONAL]
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```shell
$ litellm --add_key ANTHROPIC_API_KEY=my-api-key
```

</TabItem>
<TabItem value="perplexity" label="PerplexityAI">

```shell
$ litellm --add_key PERPLEXITYAI_API_KEY=my-api-key
```

</TabItem>

<TabItem value="together_ai" label="TogetherAI">

```shell
$ litellm --add_key TOGETHERAI_API_KEY=my-api-key
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```shell
$ litellm --add_key REPLICATE_API_KEY=my-api-key
```

</TabItem>

<TabItem value="bedrock" label="Bedrock">

```shell
$ litellm --add_key AWS_ACCESS_KEY_ID=my-key-id
$ litellm --add_key AWS_SECRET_ACCESS_KEY=my-secret-access-key
```

</TabItem>

<TabItem value="palm" label="Palm">

```shell
$ litellm --add_key PALM_API_KEY=my-palm-key
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```shell
$ litellm --add_key AZURE_API_KEY=my-api-key
$ litellm --add_key AZURE_API_BASE=my-api-base

```

</TabItem>

<TabItem value="ai21" label="AI21">

```shell
$ litellm --add_key AI21_API_KEY=my-api-key
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```shell
$ litellm --add_key COHERE_API_KEY=my-api-key
```

</TabItem>

</Tabs>

E.g.: Set api base, max tokens and temperature. 

**For that session**: 
```shell
litellm --model ollama/llama2 \
  --api_base http://localhost:11434 \
  --max_tokens 250 \
  --temperature 0.5

# OpenAI-compatible server running on http://0.0.0.0:8000
```

**Across restarts**:  
Create a file called `litellm_config.toml` and paste this in there:

```shell
[model."ollama/llama2"] # run via `litellm --model ollama/llama2`
max_tokens = 250 # set max tokens for the model 
temperature = 0.5 # set temperature for the model 
api_base = "http://localhost:11434" # set a custom api base for the model
```

&nbsp;   

Save it to the proxy with: 
```shell
$ litellm --config -f ./litellm_config.toml 
```
LiteLLM will save a copy of this file in it's package, so it can persist these settings across restarts.


**Complete Config File**

```shell
### API KEYS ### 
[keys]
# HUGGINGFACE_API_KEY="" # Uncomment to save your Hugging Face API key
# OPENAI_API_KEY="" # Uncomment to save your OpenAI API Key
# TOGETHERAI_API_KEY="" # Uncomment to save your TogetherAI API key
# NLP_CLOUD_API_KEY="" # Uncomment to save your NLP Cloud API key
# ANTHROPIC_API_KEY="" # Uncomment to save your Anthropic API key
# REPLICATE_API_KEY="" # Uncomment to save your Replicate API key
# AWS_ACCESS_KEY_ID = "" # Uncomment to save your Bedrock/Sagemaker access keys
# AWS_SECRET_ACCESS_KEY = "" # Uncomment to save your Bedrock/Sagemaker access keys

### LITELLM PARAMS ### 
[general]
# add_function_to_prompt = True # e.g: Ollama doesn't support functions, so add it to the prompt instead
# drop_params = True # drop any params not supported by the provider (e.g. Ollama)
# default_model = "gpt-4" # route all requests to this model
# fallbacks = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"] # models you want to fallback to in case completion call fails (remember: add relevant keys) 

### MODEL PARAMS ### 
[model."ollama/llama2"] # run via `litellm --model ollama/llama2`
# max_tokens = "" # set max tokens for the model 
# temperature = "" # set temperature for the model 
# api_base = "" # set a custom api base for the model

[model."ollama/llama2".prompt_template] # [OPTIONAL] LiteLLM can automatically formats the prompt - docs: https://docs.litellm.ai/docs/completion/prompt_formatting
# MODEL_SYSTEM_MESSAGE_START_TOKEN = "[INST] <<SYS>>\n" # This does not need to be a token, can be any string
# MODEL_SYSTEM_MESSAGE_END_TOKEN = "\n<</SYS>>\n [/INST]\n" # This does not need to be a token, can be any string

# MODEL_USER_MESSAGE_START_TOKEN = "[INST] " # This does not need to be a token, can be any string
# MODEL_USER_MESSAGE_END_TOKEN = " [/INST]\n" # Applies only to user messages. Can be any string.

# MODEL_ASSISTANT_MESSAGE_START_TOKEN = "" # Applies only to assistant messages. Can be any string.
# MODEL_ASSISTANT_MESSAGE_END_TOKEN = "\n" # Applies only to system messages. Can be any string.

# MODEL_PRE_PROMPT = "You are a good bot" # Applied at the start of the prompt
# MODEL_POST_PROMPT = "Now answer as best as you can" # Applied at the end of the prompt
```
[**🔥 [Tutorial] modify a model prompt on the proxy**](./tutorials/model_config_proxy.md)


### Track Costs
By default litellm proxy writes cost logs to litellm/proxy/costs.json

How can the proxy be better? Let us know [here](https://github.com/BerriAI/litellm/issues)
```json
{
  "Oct-12-2023": {
    "claude-2": {
      "cost": 0.02365918,
      "num_requests": 1
    }
  }
}
```

You can view costs on the cli using 
```shell
litellm --cost
```

## Support/ talk with founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

