import Image from '@theme/IdealImage';

# OpenAI Proxy Server

A simple, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs in the OpenAI Input/Output format

## Endpoints:
- `/chat/completions` - chat completions endpoint to call 100+ LLMs
- `/models` - available models on server

[![Deploy](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run?git_repo=https://github.com/BerriAI/litellm)

:::info
We want to learn how we can make the proxy better! Meet the [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
::: 


## Usage 

```shell 
$ git clone https://github.com/BerriAI/litellm.git
```
```shell
$ cd ./litellm/openai-proxy
```

```shell
$ uvicorn main:app --host 0.0.0.0 --port 8000
```

## Replace openai base
```python 
import openai 
openai.api_base = "http://0.0.0.0:8000" # proxy url
openai.api_key = "does-not-matter"
# call cohere
response = openai.ChatCompletion.create(
    model="command-nightly", 
    messages=[{"role":"user", "content":"Hey!"}],
    api_key="your-cohere-api-key", # enter your key here
)

# call bedrock 
response = openai.ChatCompletion.create(
    model = "bedrock/anthropic.claude-instant-v1",
    messages = [
        {
            "role": "user",
            "content": "Hey!"
        }
    ],
    aws_access_key_id="",
    aws_secret_access_key="",
    aws_region_name="us-west-2",
)

print(response)
``` 

:::info
Looking for the CLI tool/local proxy? It's [here](./proxy_server.md)
::: 

## Deploy on Google Cloud Run

[![Deploy](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run?git_repo=https://github.com/BerriAI/litellm)

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



