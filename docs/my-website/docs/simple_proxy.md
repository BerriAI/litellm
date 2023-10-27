import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 💥 LiteLLM Server - Deploy LiteLLM

A simple, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs in the OpenAI Input/Output format

[**See Code**](https://github.com/BerriAI/litellm/tree/main/litellm_server)

:::info
We want to learn how we can make the server better! Meet the [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
::: 

## Usage 

```shell
docker run -e PORT=8000 -e OPENAI_API_KEY=<your-openai-key> -p 8000:8000 ghcr.io/berriai/litellm:latest

# UVICORN: OpenAI Proxy running on http://0.0.0.0:8000
```

```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
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

If, you're calling it via Huggingface Inference Endpoints. 
```shell
$ docker run -e PORT=8000 -e HUGGINGFACE_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

Else,
```shell
$ docker run -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest
```


</TabItem>
<TabItem value="anthropic" label="Anthropic">

```shell
$ docker run -e PORT=8000 -e ANTHROPIC_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>

<TabItem value="ollama" label="Ollama">

```shell
$ docker run -e PORT=8000 -e OLLAMA_API_BASE=<your-ollama-api-base> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>


<TabItem value="together_ai" label="TogetherAI">

```shell
$ docker run -e PORT=8000 -e TOGETHERAI_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```shell
$ docker run -e PORT=8000 -e REPLICATE_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>

<TabItem value="palm" label="Palm">

```shell
$ docker run -e PORT=8000 -e PALM_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```shell
$ docker run -e PORT=8000 -e AZURE_API_KEY=<your-api-key> -e AZURE_API_BASE=<your-api-base> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>

<TabItem value="ai21" label="AI21">

```shell
$ docker run -e PORT=8000 -e AI21_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```shell
$ docker run -e PORT=8000 -e COHERE_API_KEY=<your-api-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
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

## Tutorials 

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
      - OPENAI_API_KEY=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

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
</Tabs>

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






