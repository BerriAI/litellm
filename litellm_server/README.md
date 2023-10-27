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

LiteLLM Server provides: 
- A fixed endpoint to call all LiteLLM-supported models/providers
- Caching + Logging capabilities (Redis and Langfuse, respectively)

## Usage 

```shell
docker run -e PORT=8000 -e OPENAI_API_KEY=<your-openai-key> -p 8000:8000 ghcr.io/berriai/litellm:latest
```
OpenAI Proxy running on http://0.0.0.0:8000

```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/providers)
## Endpoints:
- `/chat/completions` - chat completions endpoint to call 100+ LLMs
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

Add Langfuse Logging to your server via environment variables  

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

## Running Locally
```shell 
$ git clone https://github.com/BerriAI/litellm.git
```
```shell
$ cd ./litellm/litellm_server
```

```shell
$ uvicorn main:app --host 0.0.0.0 --port 8000
```
### Custom Config 
1. Create + Modify [router_config.yaml](https://github.com/BerriAI/litellm/blob/main/router_config_template.yaml) (save your azure/openai/etc. deployment info)
```shell
cp ./router_config_template.yaml ./router_config.yaml
```
2. Build Docker Image
```shell
docker build -t litellm_server . --build-arg CONFIG_FILE=./router_config.yaml 
```
3. Run Docker Image
```shell
docker run --name litellm_server -e PORT=8000 -p 8000:8000 litellm_server
```
