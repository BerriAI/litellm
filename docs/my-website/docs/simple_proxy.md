# OpenAI Proxy Server
A simple, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

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
openai.api_base = "http://0.0.0.0:8000"

# call cohere
openai.api_key = "my-cohere-key" # this gets passed as a header 

response = openai.ChatCompletion.create(model="command-nightly", messages=[{"role":"user", "content":"Hey!"}])

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

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/proxy_server)


:::info
Looking for the CLI tool/local proxy? It's [here](./proxy_server.md)
::: 

## Deploy on Google Cloud Run

[![Deploy](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run?git_repo=https://github.com/BerriAI/litellm)


