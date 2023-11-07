import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 💥 OpenAI Proxy Server

LiteLLM Server manages:

* Calling 100+ LLMs [Huggingface/Bedrock/TogetherAI/etc.](#other-supported-models) in the OpenAI `ChatCompletions` & `Completions` format
* Set custom prompt templates + model-specific configs (`temperature`, `max_tokens`, etc.)

## Quick Start 

```shell
$ litellm --model huggingface/bigcode/starcoder

#INFO: Proxy running on http://0.0.0.0:8000
```

### Test
In a new shell, run, this will make an `openai.ChatCompletion` request
```shell
litellm --test
```

This will now automatically route any requests for gpt-3.5-turbo to bigcode starcoder, hosted on huggingface inference endpoints. 

#### Output
```json
{
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "message": {
        "content": ", and create a new test page.\n\n### Test data\n\n- A user named",
        "role": "assistant"
      }
    }
  ],
  "id": "chatcmpl-56634359-d4ce-4dbc-972c-86a640e3a5d8",
  "created": 1699308314.054251,
  "model": "huggingface/bigcode/starcoder",
  "usage": {
    "completion_tokens": 16,
    "prompt_tokens": 10,
    "total_tokens": 26
  }
}
```

### Supported LLMs
<Tabs>
<TabItem value="bedrock" label="Bedrock">

```shell
$ export AWS_ACCESS_KEY_ID=""
$ export AWS_REGION_NAME="" # e.g. us-west-2
$ export AWS_SECRET_ACCESS_KEY=""
```

```shell
$ litellm --model bedrock/anthropic.claude-v2
```
</TabItem>
<TabItem value="huggingface" label="Huggingface (TGI)">

```shell
$ export HUGGINGFACE_API_KEY=my-api-key #[OPTIONAL]
```
```shell
$ litellm --model huggingface/<huggingface-model-name> --api_base https://<your-hf-endpoint># e.g. huggingface/mistralai/Mistral-7B-v0.1
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```shell
$ export ANTHROPIC_API_KEY=my-api-key
```
```shell
$ litellm --model claude-instant-1
```

</TabItem>
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
<TabItem value="together_ai" label="TogetherAI">

```shell
$ export TOGETHERAI_API_KEY=my-api-key
```
```shell
$ litellm --model together_ai/lmsys/vicuna-13b-v1.5-16k
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```shell
$ export REPLICATE_API_KEY=my-api-key
```
```shell
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
```
```shell
$ litellm --model palm/chat-bison
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```shell
$ export AZURE_API_KEY=my-api-key
$ export AZURE_API_BASE=my-api-base
```
```
$ litellm --model azure/my-deployment-name
```

</TabItem>

<TabItem value="ai21" label="AI21">

```shell
$ export AI21_API_KEY=my-api-key
```

```shell
$ litellm --model j2-light
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```shell
$ export COHERE_API_KEY=my-api-key
```

```shell
$ litellm --model command-nightly
```

</TabItem>

</Tabs>

### Server Endpoints
- POST `/chat/completions` - chat completions endpoint to call 100+ LLMs
- POST `/completions` - completions endpoint
- POST `/embeddings` - embedding endpoint for Azure, OpenAI, Huggingface endpoints
- GET `/models` - available models on server

## Advanced

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


### [TUTORIAL] LM-Evaluation Harness with TGI

Evaluate LLMs 20x faster with TGI via litellm proxy's `/completions` endpoint. 

This tutorial assumes you're using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

**Step 1: Start the local proxy**
```shell
$ litellm --model huggingface/bigcode/starcoder
```

OpenAI Compatible Endpoint at http://0.0.0.0:8000

**Step 2: Set OpenAI API Base**
```shell
$ export OPENAI_API_BASE="http://0.0.0.0:8000"
```

**Step 3: Run LM-Eval-Harness**

```shell
$ python3 main.py \
  --model gpt3 \
  --model_args engine=huggingface/bigcode/starcoder \
  --tasks hellaswag
```


### Caching
#### Control caching per completion request
Caching can be switched on/off per /chat/completions request
- Caching on for completion - pass `caching=True`:
  ```shell
  curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7,
     "caching": true
   }'
  ```
- Caching off for completion - pass `caching=False`:
  ```shell
  curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7,
     "caching": false
   }'
  ```

### Set Custom Prompt Templates

LiteLLM by default checks if a model has a [prompt template and applies it](./completion/prompt_formatting.md) (e.g. if a huggingface model has a saved chat template in it's tokenizer_config.json). However, you can also set a custom prompt template on your proxy in the `config.yaml`: 

**Step 1**: Save your prompt template in a `config.yaml`
```yaml
# Model-specific parameters
model_list:
  - model_name: mistral-7b # model alias
    litellm_params: # actual params for litellm.completion()
      model: "huggingface/mistralai/Mistral-7B-Instruct-v0.1" 
      api_base: "<your-api-base>"
      api_key: "<your-api-key>" # [OPTIONAL] for hf inference endpoints
      initial_prompt_value: "\n"
      roles: {"system":{"pre_message":"<|im_start|>system\n", "post_message":"<|im_end|>"}, "assistant":{"pre_message":"<|im_start|>assistant\n","post_message":"<|im_end|>"}, "user":{"pre_message":"<|im_start|>user\n","post_message":"<|im_end|>"}}
      final_prompt_value: "\n"
      bos_token: "<s>"
      eos_token: "</s>"
      max_tokens: 4096
```

**Step 2**: Start server with config

```shell
$ litellm --config /path/to/config.yaml
```

### Multiple Models 

If you have 1 model running on a local GPU and another that's hosted (e.g. on Runpod), you can call both via the same litellm server by listing them in your `config.yaml`. 

```yaml
model_list:
  - model_name: zephyr-alpha
    litellm_params: # params for litellm.completion() - https://docs.litellm.ai/docs/completion/input#input---request-body
      model: huggingface/HuggingFaceH4/zephyr-7b-alpha
      api_base: http://0.0.0.0:8001
  - model_name: zephyr-beta
    litellm_params:
      model: huggingface/HuggingFaceH4/zephyr-7b-beta
      api_base: https://<my-hosted-endpoint>
```

```shell
$ litellm --config /path/to/config.yaml
```

### Evaluate model

If you're repo let's you set model name, you can call the specific model by just passing in that model's name - 

```python
import openai 
openai.api_base = "http://0.0.0.0:8000" 

completion = openai.ChatCompletion.create(model="zephyr-alpha", messages=[{"role": "user", "content": "Hello world"}])
print(completion.choices[0].message.content)
```

If you're repo only let's you specify api base, then you can add the model name to the api base passed in - 

```python
import openai 
openai.api_base = "http://0.0.0.0:8000/openai/deployments/zephyr-alpha/chat/completions" # zephyr-alpha will be used 

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])
print(completion.choices[0].message.content)
```

### Save Model-specific params (API Base, API Keys, Temperature, etc.)
Use the [router_config_template.yaml](https://github.com/BerriAI/litellm/blob/main/router_config_template.yaml) to save model-specific information like api_base, api_key, temperature, max_tokens, etc. 

**Step 1**: Create a `config.yaml` file
```shell
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params: # params for litellm.completion() - https://docs.litellm.ai/docs/completion/input#input---request-body
      model: azure/chatgpt-v-2 # azure/<your-deployment-name>
      api_key: your_azure_api_key
      api_version: your_azure_api_version
      api_base: your_azure_api_base
  - model_name: mistral-7b
    litellm_params:
      model: ollama/mistral
      api_base: your_ollama_api_base
```

**Step 2**: Start server with config

```shell
$ litellm --config /path/to/config.yaml
```
### Model Alias 

Set a model alias for your deployments. 

In the `config.yaml` the model_name parameter is the user-facing name to use for your deployment. 

E.g.: If we want to save a Huggingface TGI Mistral-7b deployment, as 'mistral-7b' for our users, we might save it as: 

```yaml
model_list:
  - model_name: mistral-7b # ALIAS
    litellm_params:
      model: huggingface/mistralai/Mistral-7B-Instruct-v0.1 # ACTUAL NAME
      api_key: your_huggingface_api_key # [OPTIONAL] if deployed on huggingface inference endpoints
      api_base: your_api_base # url where model is deployed 
```





<!-- 
## Tutorials (Chat-UI, NeMO-Guardrails, PromptTools, Phoenix ArizeAI, Langchain, ragas, LlamaIndex, etc.)

**Start server:**
```shell
`docker run -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest`
```
The server is now live on http://0.0.0.0:8000

<Tabs>
<TabItem value="chat-ui" label="Chat UI">

Here's the `docker-compose.yml` for running LiteLLM Server with Mckay Wrigley's Chat-UI: 
```yaml
version: '3'
services:
  container1:
    image: ghcr.io/berriai/litellm:latest
    ports:
      - '8000:8000'
    environment:
      - PORT=8000
      - OPENAI_API_KEY=<your-openai-key>

  container2:
    image: ghcr.io/mckaywrigley/chatbot-ui:main
    ports:
      - '3000:3000'
    environment:
      - OPENAI_API_KEY=my-fake-key
      - OPENAI_API_HOST=http://container1:8000
```

Run this via: 
```shell
docker-compose up
```
</TabItem>
<TabItem value="nemo-guardrails" label="NeMO-Guardrails">

#### Adding NeMO-Guardrails to Bedrock 

1. Start server
```shell
`docker run -e PORT=8000 -e AWS_ACCESS_KEY_ID=<your-aws-access-key> -e AWS_SECRET_ACCESS_KEY=<your-aws-secret-key> -p 8000:8000 ghcr.io/berriai/litellm:latest`
```

2. Install dependencies
```shell
pip install nemoguardrails langchain
```

3. Run script
```python
import openai
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="bedrock/anthropic.claude-v2", openai_api_base="http://0.0.0.0:8000", openai_api_key="my-fake-key")

from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config.yml")
app = LLMRails(config, llm=llm)

new_message = app.generate(messages=[{
    "role": "user",
    "content": "Hello! What can you do for me?"
}])
``` 
</TabItem>
<TabItem value="prompttools" label="PromptTools">

Use [PromptTools](https://github.com/hegelai/prompttools) for evaluating different LLMs

1. Start server
```shell
`docker run -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest`
```

2. Install dependencies 
```python 
pip install prompttools
```

3. Run script 
```python 
import os
os.environ['DEBUG']=""  # Set this to "" to call OpenAI's API
os.environ['AZURE_OPENAI_KEY'] = "my-api-key"  # Insert your key here

from typing import Dict, List
from prompttools.experiment import OpenAIChatExperiment

models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]
messages = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who was the first president?"},
    ]
]
temperatures = [0.0, 1.0]
# You can add more parameters that you'd like to test here.

experiment = OpenAIChatExperiment(models, messages, temperature=temperatures, azure_openai_service_configs={"AZURE_OPENAI_ENDPOINT": "http://0.0.0.0:8000", "API_TYPE": "azure", "API_VERSION": "2023-05-15"})
```
</TabItem>
<TabItem value="phoenix-arizeai" label="ArizeAI">

Use [Arize AI's LLM Evals](https://github.com/Arize-ai/phoenix#llm-evals) to evaluate different LLMs

1. Start server
```shell
`docker run -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest`
```

2. Use this LLM Evals Quickstart colab
[![Open in Colab](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=grey&color=blue&logoColor=orange&label=%20)](https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/evals/evaluate_relevance_classifications.ipynb)

3. Call the model
```python
import openai 

## SET API BASE + PROVIDER KEY
openai.api_base = "http://0.0.0.0:8000
openai.api_key = "my-anthropic-key"

## CALL MODEL 
model = OpenAIModel(
    model_name="claude-2",
    temperature=0.0,
)
```
</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(model_name="claude-instant-1", openai_api_key="my-anthropic-key", openai_api_base="http://0.0.0.0:8000")

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
chat(messages)
```
</TabItem>
<TabItem value="ragas" label="ragas">

#### Evaluating with Open-Source LLMs 

Use [Ragas](https://github.com/explodinggradients/ragas/blob/7b123533df80d0ada33a2cb2dd2fdedf36807f33/docs/howtos/customisations/llms.ipynb#L247) to evaluate LLMs for RAG-scenarios.
```python
from langchain.chat_models import ChatOpenAI

inference_server_url = "http://localhost:8080/v1"

chat = ChatOpenAI(
    model="bedrock/anthropic.claude-v2",
    openai_api_key="no-key",
    openai_api_base=inference_server_url,
    max_tokens=5,
    temperature=0,
)

from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)
from ragas.metrics.critique import harmfulness

# change the LLM

faithfulness.llm.langchain_llm = chat
answer_relevancy.llm.langchain_llm = chat
context_precision.llm.langchain_llm = chat
context_recall.llm.langchain_llm = chat
harmfulness.llm.langchain_llm = chat


# evaluate
from ragas import evaluate

result = evaluate(
    fiqa_eval["baseline"].select(range(5)),  # showing only 5 for demonstration
    metrics=[faithfulness],
)

result
```
</TabItem>
<TabItem value="llama_index" label="Llama Index">

```python
!pip install llama-index
```
```python
from llama_index.llms import OpenAI

response = OpenAI(model="claude-2", api_key="your-anthropic-key",api_base="http://0.0.0.0:8000").complete('Paul Graham is ')
print(response)
```
</TabItem>
</Tabs> -->

