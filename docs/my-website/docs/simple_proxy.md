import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 💥 Evaluate LLMs - OpenAI Compatible Server

LiteLLM Server, is a simple, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs in the OpenAI Input/Output format

LiteLLM Server supports:

* LLM API Calls in the OpenAI ChatCompletions format
* Caching + Logging capabilities (Redis and Langfuse, respectively)
* Setting API keys in the request headers or in the .env

[**See Code**](https://github.com/BerriAI/litellm/tree/main/litellm_server)

:::info
We want to learn how we can make the server better! Meet the [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
::: 

## Usage 

```shell
docker run -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest
```
OpenAI Proxy running on http://0.0.0.0:8000

```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $YOUR_API_KEY"
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

#### Other supported models:
<Tabs>
<TabItem value="bedrock" label="Bedrock">

```shell
$ docker run -e PORT=8000 -e AWS_ACCESS_KEY_ID=<your-access-key> -e AWS_SECRET_ACCESS_KEY=<your-secret-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>
<TabItem value="huggingface" label="Huggingface">

**Set API Keys in .env**  
If, you're calling it via Huggingface Inference Endpoints. 
```shell
$ docker run -e PORT=8000 -e HUGGINGFACE_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

Else,
```shell
$ docker run -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest
```

**Set API Keys in request headers**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HUGGINGFACE_API_KEY"
  -d '{
     "model": "huggingface/bigcoder/starcoder",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```


</TabItem>
<TabItem value="anthropic" label="Anthropic">

**Set API Keys in .env**  
```shell
$ docker run -e PORT=8000 -e ANTHROPIC_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

**Set API Keys in request headers**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ANTHROPIC_API_KEY"
  -d '{
     "model": "claude-2",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```


</TabItem>

<TabItem value="ollama" label="Ollama">

```shell
$ docker run -e PORT=8000 -e OLLAMA_API_BASE=<your-ollama-api-base> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>


<TabItem value="together_ai" label="TogetherAI">

**Set API Keys in .env**  
```shell
$ docker run -e PORT=8000 -e TOGETHERAI_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

**Set API Keys in request headers**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOGETHERAI_API_KEY"
  -d '{
     "model": "together_ai/togethercomputer/llama-2-70b-chat",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

</TabItem>

<TabItem value="replicate" label="Replicate">

**Set API Keys in .env**  
```shell
$ docker run -e PORT=8000 -e REPLICATE_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

**Set API Keys in request headers**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $REPLICATE_API_KEY"
  -d '{
     "model": "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```


</TabItem>

<TabItem value="palm" label="Palm">

**Set API Keys in .env**  
```shell
$ docker run -e PORT=8000 -e PALM_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

**Set API Keys in request headers**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PALM_API_KEY"
  -d '{
     "model": "palm/chat-bison",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

**Set API Keys in .env**  
```shell
$ docker run -e PORT=8000 -e AZURE_API_KEY=<your-api-key> -e AZURE_API_BASE=<your-api-base> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>

<TabItem value="ai21" label="AI21">

**Set API Keys in .env**  
```shell
$ docker run -e PORT=8000 -e AI21_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

**Set API Keys in request headers**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AI21_API_KEY"
  -d '{
     "model": "j2-mid",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

</TabItem>

<TabItem value="cohere" label="Cohere">

**Set API Keys in .env**  

```shell
$ docker run -e PORT=8000 -e COHERE_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

**Set API Keys in request headers**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $COHERE_API_KEY"
  -d '{
     "model": "command-nightly",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```


</TabItem>

</Tabs>

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
</Tabs>


## Endpoints:
- `/chat/completions` - chat completions endpoint to call 100+ LLMs
- `/embeddings` - embedding endpoint for Azure, OpenAI, Huggingface endpoints
- `/models` - available models on server


## Save Model-specific params (API Base, API Keys, Temperature, etc.)
Use the [router_config_template.yaml](https://github.com/BerriAI/litellm/blob/main/router_config_template.yaml) to save model-specific information like api_base, api_key, temperature, max_tokens, etc. 

1. Create a `config.yaml` file
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

2. Start the server

```shell
docker run -e PORT=8000 -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml ghcr.io/berriai/litellm:latest
```
## Model Alias 

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

## Caching 

Add Redis Caching to your server via environment variables  

```env
### REDIS
REDIS_HOST = "" 
REDIS_PORT = "" 
REDIS_PASSWORD = "" 
```

Docker command: 

```shell
docker run -e REDIST_HOST=<your-redis-host> -e REDIS_PORT=<your-redis-port> -e REDIS_PASSWORD=<your-redis-password> -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest
```

## Logging 

1. Debug Logs
Print the input/output params by setting `SET_VERBOSE = "True"`.

Docker command:

```shell
docker run -e SET_VERBOSE="True" -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest
```
2. Add Langfuse Logging to your server via environment variables  

```env
### LANGFUSE
LANGFUSE_PUBLIC_KEY = ""
LANGFUSE_SECRET_KEY = ""
# Optional, defaults to https://cloud.langfuse.com
LANGFUSE_HOST = "" # optional
```

Docker command: 

```shell
docker run -e LANGFUSE_PUBLIC_KEY=<your-public-key> -e LANGFUSE_SECRET_KEY=<your-secret-key> -e LANGFUSE_HOST=<your-langfuse-host> -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest
```

## Local Usage 

```shell 
$ git clone https://github.com/BerriAI/litellm.git
```
```shell
$ cd ./litellm/litellm_server
```

```shell
$ uvicorn main:app --host 0.0.0.0 --port 8000
```

## Setting LLM API keys
This server allows two ways of passing API keys to litellm
- Environment Variables - This server by default assumes the LLM API Keys are stored in the environment variables
- Dynamic Variables passed to `/chat/completions`
  - Set `AUTH_STRATEGY=DYNAMIC` in the Environment 
  - Pass required auth params `api_key`,`api_base`, `api_version` with the request params


<Tabs>
<TabItem value="gcp-run" label="Google Cloud Run">

## Deploy on Google Cloud Run
**Click the button** to deploy to Google Cloud Run

[![Deploy](https://deploy.cloud.run/button.svg)](https://l.linklyhq.com/l/1uHtX)

On a successfull deploy your Cloud Run Shell will have this output
<Image img={require('../img/cloud_run0.png')} />

### Testing your deployed server
**Assuming the required keys are set as Environment Variables**

https://litellm-7yjrj3ha2q-uc.a.run.app is our example server, substitute it with your deployed cloud run app

<Tabs>
<TabItem value="openai" label="OpenAI">

```shell
curl https://litellm-7yjrj3ha2q-uc.a.run.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

</TabItem>
<TabItem value="azure" label="Azure">

```shell
curl https://litellm-7yjrj3ha2q-uc.a.run.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "azure/<your-deployment-name>",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

</TabItem>

<TabItem value="anthropic" label="Anthropic">

```shell
curl https://litellm-7yjrj3ha2q-uc.a.run.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "claude-2",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7,
   }'
```
</TabItem>

</Tabs>

### Set LLM API Keys
#### Environment Variables 
More info [here](https://cloud.google.com/run/docs/configuring/services/environment-variables#console)

1. In the Google Cloud console, go to Cloud Run: [Go to Cloud Run](https://console.cloud.google.com/run)

2. Click on the **litellm** service
<Image img={require('../img/cloud_run1.png')} />

3. Click **Edit and Deploy New Revision**
<Image img={require('../img/cloud_run2.png')} />

4. Enter your Environment Variables
Example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
<Image img={require('../img/cloud_run3.png')} />

</TabItem>
<TabItem value="render" label="Render">

## Deploy on Render
**Click the button** to deploy to Render

[![Deploy](https://render.com/images/deploy-to-render-button.svg)](https://l.linklyhq.com/l/1uHsr)

On a successfull deploy https://dashboard.render.com/ should display the following
<Image img={require('../img/render1.png')} />

<Image img={require('../img/render2.png')} />
</TabItem>
<TabItem value="aws-apprunner" label="AWS Apprunner">

## Deploy on AWS Apprunner
1. Fork LiteLLM https://github.com/BerriAI/litellm 
2. Navigate to to App Runner on AWS Console: https://console.aws.amazon.com/apprunner/home#/services
3. Follow the steps in the video below
<iframe width="800" height="450" src="https://www.loom.com/embed/5fccced4dde8461a8caeee97addb2231?sid=eac60660-073e-455e-a737-b3d05a5a756a" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

4. Testing your deployed endpoint

  **Assuming the required keys are set as Environment Variables** Example: `OPENAI_API_KEY`

  https://b2w6emmkzp.us-east-1.awsapprunner.com is our example server, substitute it with your deployed apprunner endpoint

  <Tabs>
  <TabItem value="openai" label="OpenAI">

  ```shell
  curl https://b2w6emmkzp.us-east-1.awsapprunner.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-3.5-turbo",
      "messages": [{"role": "user", "content": "Say this is a test!"}],
      "temperature": 0.7
    }'
  ```

  </TabItem>
  <TabItem value="azure" label="Azure">

  ```shell
  curl https://b2w6emmkzp.us-east-1.awsapprunner.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "azure/<your-deployment-name>",
      "messages": [{"role": "user", "content": "Say this is a test!"}],
      "temperature": 0.7
    }'
  ```

  </TabItem>

  <TabItem value="anthropic" label="Anthropic">

  ```shell
  curl https://b2w6emmkzp.us-east-1.awsapprunner.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "claude-2",
      "messages": [{"role": "user", "content": "Say this is a test!"}],
      "temperature": 0.7,
    }'
  ```
  </TabItem>

  </Tabs>

</TabItem>
</Tabs>

## Advanced
### Caching - Completion() and Embedding() Responses

Enable caching by adding the following credentials to your server environment

  ```
  REDIS_HOST = ""       # REDIS_HOST='redis-18841.c274.us-east-1-3.ec2.cloud.redislabs.com'
  REDIS_PORT = ""       # REDIS_PORT='18841'
  REDIS_PASSWORD = ""   # REDIS_PASSWORD='liteLlmIsAmazing'
  ```

#### Test Caching 
Send the same request twice:
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7
   }'

curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7
   }'
```

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






