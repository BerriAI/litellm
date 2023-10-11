# LiteLLM/Ollama Docker Image 
[![Chat on WhatsApp](https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square)](https://wa.link/huol9n) [![Chat on Discord](https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square)](https://discord.gg/wuPM9dRgDw) 
## An OpenAI API compatible server for local LLMs - llama2, mistral, codellama

## Quick Start:
Docker Hub: 
https://hub.docker.com/repository/docker/litellm/ollama/general

```shell
docker pull litellm/ollama
```

```shell
docker run --name ollama litellm/ollama
```

### Test the server container
On the docker container run the `test.py` file using `python3 test.py`


## Supported Models

| Model              | Parameters | Size  | Download                       |
| ------------------ | ---------- | ----- | ------------------------------ |
| Mistral            | 7B         | 4.1GB | `ollama run mistral`           |
| Llama 2            | 7B         | 3.8GB | `ollama run llama2`            |
| Code Llama         | 7B         | 3.8GB | `ollama run codellama`         |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama run llama2-uncensored` |
| Llama 2 13B        | 13B        | 7.3GB | `ollama run llama2:13b`        |
| Llama 2 70B        | 70B        | 39GB  | `ollama run llama2:70b`        |
| Orca Mini          | 3B         | 1.9GB | `ollama run orca-mini`         |
| Vicuna             | 7B         | 3.8GB | `ollama run vicuna`            |


## Making a request to this server
```python
import openai

api_base = f"http://0.0.0.0:8000" # base url for server

openai.api_base = api_base
openai.api_key = "temp-key"
print(openai.api_base)


print(f'LiteLLM: response from proxy with streaming')
response = openai.ChatCompletion.create(
    model="ollama/llama2", 
    messages = [
        {
            "role": "user",
            "content": "this is a test request, acknowledge that you got it"
        }
    ],
    stream=True
)

for chunk in response:
    print(f'LiteLLM: streaming response from proxy {chunk}')
```

## Responses from this server 
```json
{
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": " Hello! I acknowledge receipt of your test request. Please let me know if there's anything else I can assist you with.",
        "role": "assistant",
        "logprobs": null
      }
    }
  ],
  "id": "chatcmpl-403d5a85-2631-4233-92cb-01e6dffc3c39",
  "created": 1696992706.619709,
  "model": "ollama/llama2",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 25,
    "total_tokens": 43
  }
}
```

# Support / talk with founders
- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai