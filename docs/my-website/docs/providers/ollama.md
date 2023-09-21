# Ollama 
LiteLLM supports all models from [Ollama](https://github.com/jmorganca/ollama)

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
### Ollama Models
Ollama supported models: https://github.com/jmorganca/ollama

| Model Name           | Function Call                                                                     | Required OS Variables          |
|----------------------|-----------------------------------------------------------------------------------|--------------------------------|
| Llama2 7B            | `completion(model='ollama/llama2', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
| Llama2 13B           | `completion(model='ollama/llama2:13b', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
| Llama2 70B           | `completion(model='ollama/llama2:70b', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
| Llama2 Uncensored    | `completion(model='ollama/llama2-uncensored', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
| Orca Mini            | `completion(model='ollama/orca-mini', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
| Vicuna               | `completion(model='ollama/vicuna', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
| Nous-Hermes          | `completion(model='ollama/nous-hermes', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
| Nous-Hermes 13B     | `completion(model='ollama/nous-hermes:13b', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
| Wizard Vicuna Uncensored | `completion(model='ollama/wizard-vicuna', messages, api_base="http://localhost:11434", stream=True)` | No API Key required |
