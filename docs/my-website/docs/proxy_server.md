import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [OLD PROXY 👉 [**NEW** proxy here](./simple_proxy)] Local OpenAI Proxy Server

A fast, and lightweight OpenAI-compatible server to call 100+ LLM APIs. 

:::info

Docs outdated. New docs 👉 [here](./simple_proxy)

:::

## Usage 
```shell
pip install 'litellm[proxy]'
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

### Tutorial: Use with Multiple LLMs + LibreChat/Chatbot-UI/Auto-Gen/ChatDev/Langroid,etc. 
<Tabs>
<TabItem value="multiple-LLMs" label="Multiple LLMs">

Replace openai base: 
```python
import openai 

openai.api_key = "any-string-here"
openai.api_base = "http://0.0.0.0:8080" # your proxy url

# call openai
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey"}])

print(response)

# call cohere
response = openai.ChatCompletion.create(model="command-nightly", messages=[{"role": "user", "content": "Hey"}])

print(response)
```
</TabItem>
<TabItem value="librechat" label="LibreChat">

#### 1. Clone the repo

```shell
git clone https://github.com/danny-avila/LibreChat.git
```


#### 2. Modify `docker-compose.yml`
```yaml
OPENAI_REVERSE_PROXY=http://host.docker.internal:8000/v1/chat/completions
```

#### 3. Save fake OpenAI key in `.env`
```env
OPENAI_API_KEY=sk-1234
```

#### 4. Run LibreChat: 
```shell
docker compose up
```
</TabItem>
<TabItem value="smart-chatbot-ui" label="SmartChatbotUI">

#### 1. Clone the repo
```shell
git clone https://github.com/dotneet/smart-chatbot-ui.git
```

#### 2. Install Dependencies
```shell
npm i
```

#### 3. Create your env
```shell
cp .env.local.example .env.local
```

#### 4. Set the API Key and Base
```env
OPENAI_API_KEY="my-fake-key"
OPENAI_API_HOST="http://0.0.0.0:8000
```

#### 5. Run with docker compose
```shell
docker compose up -d
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
        "api_base": "http://0.0.0.0:8000",  #litellm compatible endpoint
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
<TabItem value="multi-LLM AutoGen" label="AutoGen Multi-LLM">


```python
from autogen import AssistantAgent, GroupChatManager, UserProxyAgent
from autogen.agentchat import GroupChat
config_list = [
    {
        "model": "ollama/mistralorca",
        "api_base": "http://0.0.0.0:8000",  # litellm compatible endpoint
        "api_type": "open_ai",
        "api_key": "NULL",  # just a placeholder
    }
]
llm_config = {"config_list": config_list, "seed": 42}

code_config_list = [
    {
        "model": "ollama/phind-code",
        "api_base": "http://0.0.0.0:8000",  # litellm compatible endpoint
        "api_type": "open_ai",
        "api_key": "NULL",  # just a placeholder
    }
]

code_config = {"config_list": code_config_list, "seed": 42}

admin = UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    llm_config=llm_config,
    code_execution_config=False,
)


engineer = AssistantAgent(
    name="Engineer",
    llm_config=code_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)
planner = AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=llm_config,
)
executor = UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    llm_config=llm_config,
    code_execution_config={"last_n_messages": 3, "work_dir": "paper"},
)
critic = AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=llm_config,
)
groupchat = GroupChat(
    agents=[admin, engineer, planner, executor, critic],
    messages=[],
    max_round=50,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)


admin.initiate_chat(
    manager,
    message="""
""",
)
```

Credits [@Nathan](https://gist.github.com/CUexter) for this tutorial.
</TabItem>
<TabItem value="chatDev" label="ChatDev">

### Setup ChatDev ([Docs](https://github.com/OpenBMB/ChatDev#%EF%B8%8F-quickstart))
```shell
git clone https://github.com/OpenBMB/ChatDev.git
cd ChatDev
conda create -n ChatDev_conda_env python=3.9 -y
conda activate ChatDev_conda_env
pip install -r requirements.txt
```
### Run ChatDev w/ Proxy
```shell 
export OPENAI_API_KEY="sk-1234"
```

```shell 
export OPENAI_API_BASE="http://0.0.0.0:8000"
```
```shell
python3 run.py --task "a script that says hello world" --name "hello world"
```
</TabItem>
<TabItem value="langroid" label="Langroid">

```python
pip install langroid
```

```python
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIGPT

# configure the LLM
my_llm_config = OpenAIGPTConfig(
    # where proxy server is listening 
    api_base="http://0.0.0.0:8000", 
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
</Tabs>

## Local Proxy

Here's how to use the local proxy to test codellama/mistral/etc. models for different github repos 

```shell
pip install litellm
```

```shell
$ ollama pull codellama # OUR Local CodeLlama  

$ litellm --model ollama/codellama --temperature 0.3 --max_tokens 2048
```

### Tutorial: Use with Multiple LLMs + Aider/AutoGen/Langroid/etc.
<Tabs>
<TabItem value="multiple-LLMs" label="Multiple LLMs">

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

</TabItem>
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
<TabItem value="multi-LLM AutoGen" label="AutoGen Multi-LLM">


```python
from autogen import AssistantAgent, GroupChatManager, UserProxyAgent
from autogen.agentchat import GroupChat
config_list = [
    {
        "model": "ollama/mistralorca",
        "api_base": "http://localhost:8000",  # litellm compatible endpoint
        "api_type": "open_ai",
        "api_key": "NULL",  # just a placeholder
    }
]
llm_config = {"config_list": config_list, "seed": 42}

code_config_list = [
    {
        "model": "ollama/phind-code",
        "api_base": "http://localhost:8000",  # litellm compatible endpoint
        "api_type": "open_ai",
        "api_key": "NULL",  # just a placeholder
    }
]

code_config = {"config_list": code_config_list, "seed": 42}

admin = UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    llm_config=llm_config,
    code_execution_config=False,
)


engineer = AssistantAgent(
    name="Engineer",
    llm_config=code_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)
planner = AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=llm_config,
)
executor = UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    llm_config=llm_config,
    code_execution_config={"last_n_messages": 3, "work_dir": "paper"},
)
critic = AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=llm_config,
)
groupchat = GroupChat(
    agents=[admin, engineer, planner, executor, critic],
    messages=[],
    max_round=50,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)


admin.initiate_chat(
    manager,
    message="""
""",
)
```

Credits [@Nathan](https://gist.github.com/CUexter) for this tutorial.
</TabItem>
<TabItem value="chatDev" label="ChatDev">

### Setup ChatDev ([Docs](https://github.com/OpenBMB/ChatDev#%EF%B8%8F-quickstart))
```shell
git clone https://github.com/OpenBMB/ChatDev.git
cd ChatDev
conda create -n ChatDev_conda_env python=3.9 -y
conda activate ChatDev_conda_env
pip install -r requirements.txt
```
### Run ChatDev w/ Proxy
```shell 
export OPENAI_API_KEY="sk-1234"
```

```shell 
export OPENAI_API_BASE="http://0.0.0.0:8000"
```
```shell
python3 run.py --task "a script that says hello world" --name "hello world"
```
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

### Logs

```shell
$ litellm --logs
```

This will return the most recent log (the call that went to the LLM API + the received response).

All logs are saved to a file called `api_logs.json` in the current directory. 

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

### Performance

We load-tested 500,000 HTTP connections on the FastAPI server for 1 minute, using [wrk](https://github.com/wg/wrk).

There are our results: 

```shell
Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   156.38ms   25.52ms 361.91ms   84.73%
    Req/Sec    13.61      5.13    40.00     57.50%
  383625 requests in 1.00m, 391.10MB read
  Socket errors: connect 0, read 1632, write 1, timeout 0
```


## Support/ talk with founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

