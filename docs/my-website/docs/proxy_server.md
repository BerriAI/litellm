# OpenAI Proxy Server

Use this to spin up a proxy api to translate openai api calls to any non-openai model (e.g. Huggingface, TogetherAI, Ollama, etc.)

This works for async + streaming as well. 

**Requirements** Make sure relevant keys are set in the local .env. [Provider List](https://docs.litellm.ai/docs/providers)
## usage
```shell 
pip install litellm
```

```shell
litellm --model <your-model-name>
```

This will host a local proxy api at : **http://localhost:8000**

[**Jump to Code**](https://github.com/BerriAI/litellm/blob/fef4146396d5d87006259e00095a62e3900d6bb4/litellm/proxy.py#L36)

## [Advanced] setting api base
If your model is running locally or on a custom endpoint

Pass in the api_base as well

```shell
litellm --model huggingface/meta-llama/llama2 --api_base https://my-endpoint.huggingface.cloud
```

## test it 

```curl 
curl --location 'http://0.0.0.0:8000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user", 
      "content": "what do you know?"
    }
  ], 
}'
```

