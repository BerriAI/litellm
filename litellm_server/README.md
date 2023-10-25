# litellm-server

A simple, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

<p align="center" style="margin: 2%">
        <a href="https://l.linklyhq.com/l/1uHsr" target="_blank">
                <img src="https://render.com/images/deploy-to-render-button.svg" width="173"/>
        </a>
        <a href="https://l.linklyhq.com/l/1uHtX" target="_blank">
                <img src="https://deploy.cloud.run/button.svg" width="200"/>
        </a>
</p>

## Usage 

```shell
docker run -e PORT=8000 -p 8000:8000 ghcr.io/berriai/litellm:latest

# UVICORN: OpenAI Proxy running on http://0.0.0.0:8000
```

## Endpoints:
- `/chat/completions` - chat completions endpoint to call 100+ LLMs
- `/router/completions` - for multiple deployments of the same model (e.g. Azure OpenAI), uses the least used deployment. [Learn more](https://docs.litellm.ai/docs/routing)
- `/models` - available models on server

## Making Requests to Proxy
### Curl

**Call OpenAI**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```
**Call Bedrock**
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "bedrock/anthropic.claude-instant-v1",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
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

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/simple_proxy)
