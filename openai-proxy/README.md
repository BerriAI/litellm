# Openai-proxy

A simple, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

<p align="center" style="margin: 2%">
        <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank">
                <img src="https://render.com/images/deploy-to-render-button.svg" width="173"/>
        </a>
        <a href="https://deploy.cloud.run" target="_blank">
                <img src="https://deploy.cloud.run/button.svg" width="200"/>
        </a>
</p>

## Usage 

```shell
docker run -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest
```

### Running Locally
```shell 
$ git clone https://github.com/BerriAI/litellm.git
```
```shell
$ cd ./litellm/openai-proxy
```

```shell
$ uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints:
- `/chat/completions` - chat completions endpoint to call 100+ LLMs
- `/models` - available models on server

## Making Requests to Proxy
### Curl
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

### Replace openai base
```python 
import openai 
openai.api_base = "http://0.0.0.0:8000"

# cohere call
response = openai.ChatCompletion.create(
        model="command-nightly",
        messages=[{"role":"user", "content":"Say this is a test!"}],
        api_key = "your-cohere-api-key"
)

# bedrock call
response = openai.ChatCompletion.create(
        model = "bedrock/anthropic.claude-instant-v1",
        messages=[{"role":"user", "content":"Say this is a test!"}],
        aws_access_key_id="",
        aws_secret_access_key="",
        aws_region_name="us-west-2",
)

print(response)
``` 

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/simple_proxy)
