import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 💥 LiteLLM Server - Deploy LiteLLM

A simple, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs in the OpenAI Input/Output format

## Endpoints:
- `/chat/completions` - chat completions endpoint to call 100+ LLMs
- `/models` - available models on server

[![Deploy](https://deploy.cloud.run/button.svg)](https://l.linklyhq.com/l/1uHtX)
[![Deploy](https://render.com/images/deploy-to-render-button.svg)](https://l.linklyhq.com/l/1uHsr)
[![Deploy](../img/deploy-to-aws.png)](https://docs.litellm.ai/docs/simple_proxy#deploy-on-aws-apprunner)

:::info
We want to learn how we can make the server better! Meet the [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
::: 


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

### Test Request
Ensure your API keys are set in the Environment for these requests

<Tabs>
<TabItem value="openai" label="OpenAI">

```shell
curl http://0.0.0.0:8000/v1/chat/completions \
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
curl http://0.0.0.0:8000/v1/chat/completions \
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
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "claude-2",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7,
   }'
```
</TabItem>

</Tabs>


## Setting LLM API keys
This server allows two ways of passing API keys to litellm
- Environment Variables - This server by default assumes the LLM API Keys are stored in the environment variables
- Dynamic Variables passed to `/chat/completions`
  - Set `AUTH_STRATEGY=DYNAMIC` in the Environment 
  - Pass required auth params `api_key`,`api_base`, `api_version` with the request params

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


## Deploy on Render
**Click the button** to deploy to Render

[![Deploy](https://render.com/images/deploy-to-render-button.svg)](https://l.linklyhq.com/l/1uHsr)

On a successfull deploy https://dashboard.render.com/ should display the following
<Image img={require('../img/render1.png')} />

<Image img={require('../img/render2.png')} />

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






