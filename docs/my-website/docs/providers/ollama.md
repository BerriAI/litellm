# Ollama 
LiteLLM supports all models from [Ollama](https://github.com/jmorganca/ollama)

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_Ollama.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Pre-requisites
Ensure you have your ollama server running

## Example usage
```python
from litellm import completion

response = completion(
    model="ollama/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434"
)
print(response)

```

## Example usage - Streaming
```python
from litellm import completion

response = completion(
    model="ollama/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434",
    stream=True
)
print(response)
for chunk in response:
    print(chunk['choices'][0]['delta'])

```

## Example usage - Streaming + Acompletion
Ensure you have async_generator installed for using ollama acompletion with streaming
```shell
pip install async_generator
```

```python
async def async_ollama():
    response = await litellm.acompletion(
        model="ollama/llama2", 
        messages=[{ "content": "what's the weather" ,"role": "user"}], 
        api_base="http://localhost:11434", 
        stream=True
    )
    async for chunk in response:
        print(chunk)

# call async_ollama
import asyncio
asyncio.run(async_ollama())

```
## Ollama Models
Ollama supported models: https://github.com/jmorganca/ollama

| Model Name           | Function Call                                                                     |
|----------------------|-----------------------------------------------------------------------------------
| Mistral    | `completion(model='ollama/mistral', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 7B            | `completion(model='ollama/llama2', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 13B           | `completion(model='ollama/llama2:13b', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 70B           | `completion(model='ollama/llama2:70b', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 Uncensored    | `completion(model='ollama/llama2-uncensored', messages, api_base="http://localhost:11434", stream=True)` | 
| Code Llama    | `completion(model='ollama/codellama', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 Uncensored    | `completion(model='ollama/llama2-uncensored', messages, api_base="http://localhost:11434", stream=True)` | 
| Orca Mini            | `completion(model='ollama/orca-mini', messages, api_base="http://localhost:11434", stream=True)` |
| Vicuna               | `completion(model='ollama/vicuna', messages, api_base="http://localhost:11434", stream=True)` |
| Nous-Hermes          | `completion(model='ollama/nous-hermes', messages, api_base="http://localhost:11434", stream=True)` |
| Nous-Hermes 13B     | `completion(model='ollama/nous-hermes:13b', messages, api_base="http://localhost:11434", stream=True)` | 
| Wizard Vicuna Uncensored | `completion(model='ollama/wizard-vicuna', messages, api_base="http://localhost:11434", stream=True)` |



## LiteLLM/Ollama Docker Image 

For Ollama LiteLLM Provides a Docker Image for an OpenAI API compatible server for local LLMs - llama2, mistral, codellama


[![Chat on WhatsApp](https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square)](https://wa.link/huol9n) [![Chat on Discord](https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square)](https://discord.gg/wuPM9dRgDw) 
### An OpenAI API compatible server for local LLMs - llama2, mistral, codellama

### Quick Start:
Docker Hub: 
For ARM Processors: https://hub.docker.com/repository/docker/litellm/ollama/general
For Intel/AMD Processors: to be added
```shell
docker pull litellm/ollama
```

```shell
docker run --name ollama litellm/ollama
```

#### Test the server container
On the docker container run the `test.py` file using `python3 test.py`


### Making a request to this server
```python
import openai

api_base = f"http://0.0.0.0:8000" # base url for server

openai.api_base = api_base
openai.api_key = "temp-key"
print(openai.api_base)


print(f'LiteLLM: response from proxy with streaming')
response = openai.chat.completions.create(
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

### Responses from this server 
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

## Support / talk with founders
- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
