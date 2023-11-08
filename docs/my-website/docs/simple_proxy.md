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

### Replace openai base

```python
import openai 

openai.api_base = "http://0.0.0.0:8000"

print(openai.ChatCompletion.create(model="test", messages=[{"role":"user", "content":"Hey!"}]))
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

## Usage

### Using with OpenAI compatible projects
LiteLLM allows you to set `openai.api_base` to the proxy server and use all LiteLLM supported LLMs in any OpenAI supported project

<Tabs>
<TabItem value="lm-harness" label="LM-Harness Evals">
This tutorial assumes you're using the `big-refactor` branch of LM Harness https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor

**Step 1: Start the local proxy**
```shell
$ litellm --model huggingface/bigcode/starcoder
```

Using a custom api base

```shell
$ export HUGGINGFACE_API_KEY=my-api-key #[OPTIONAL]
$ litellm --model huggingface/tinyllama --api_base https://k58ory32yinf1ly0.us-east-1.aws.endpoints.huggingface.cloud
```

OpenAI Compatible Endpoint at http://0.0.0.0:8000

**Step 2: Set OpenAI API Base & Key**
```shell
$ export OPENAI_API_BASE=http://0.0.0.0:8000
```

LM Harness requires you to set an OpenAI API key `OPENAI_API_SECRET_KEY` for running benchmarks
```shell
export OPENAI_API_SECRET_KEY=anything
```

**Step 3: Run LM-Eval-Harness**

```shell
python3 -m lm_eval \
  --model openai-completions \
  --model_args engine=davinci \
  --task crows_pairs_english_age

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

## Advanced



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

### Using Multiple Models 

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

