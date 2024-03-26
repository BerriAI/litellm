# LiteLLM Docker Proxy

LiteLLM Proxy is a powerful proxy server that allows you to call over 100 Large Language Models (LLMs) through a unified interface, while also providing features to track spending and set budgets for virtual keys or users.

<div align="center">

 | [日本語](README.Proxy.JP.md) | [English](README.Proxy.md) |

</div>

## Key Features

- **Unified Interface**: Seamlessly call more than 100 LLMs from providers such as Huggingface, Bedrock, and TogetherAI using the familiar OpenAI `ChatCompletions` and `Completions` format.
- **Cost Tracking**: Easily authenticate, track expenses, and set budgets using virtual keys.
- **Load Balancing**: Efficiently distribute the workload between multiple models and deployments of the same model. LiteLLM proxy can handle over 1,500 requests per second during load tests.

## Quick Start

To quickly get started with LiteLLM Proxy, you can use the CLI:

```bash
$ pip install 'litellm[proxy]'
```

If you prefer using Docker, you can start the LiteLLM container with the following command:

```bash
docker-compose -f docker\docker-compose.gemi.yml up --build
```

Upon successful startup of the container, you will see logs similar to the following:

```
litellm-1  | INFO:     Started server process [1]
litellm-1  | INFO:     Waiting for application startup.
litellm-1  |
litellm-1  | #------------------------------------------------------------#
litellm-1  | #                                                            #
litellm-1  | #              'I don't like how this works...'               #
litellm-1  | #        https://github.com/BerriAI/litellm/issues/new        #
litellm-1  | #                                                            #
litellm-1  | #------------------------------------------------------------#
litellm-1  |
litellm-1  |  Thank you for using LiteLLM! - Krrish & Ishaan
litellm-1  |
litellm-1  |
litellm-1  |
litellm-1  | Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
litellm-1  |
litellm-1  |
litellm-1  | INFO:     Application startup complete.
litellm-1  | INFO:     Uvicorn running on http://0.0.0.0:4000 (Press CTRL+C to quit)
```

To run the demo script, use the following command:

```bash
python docker\demo\demo_openai.py
```

`docker\demo\demo_openai.py`
```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://localhost:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)
print("-------------")
print(response.choices[0].message.content)
```

The output of the demo script will be displayed as follows:

```
ChatCompletion(id='chatcmpl-7ef51102-505c-4c54-9e5a-783f6d4d0401', choices=[Choice(finish_reason='stop', index=1, logprobs=None, message=ChatCompletionMessage(content="In realms of words, a dance takes place,\nA symphony of rhythm, grace.\nEach syllable a note so fine,\nWeaving stories, making hearts entwine.\n\nFrom whispers soft to thunder's roar,\nWords paint worlds, forevermore.\nThey evoke emotions, deep and true,\nGuiding us through life's every hue.", role='assistant', function_call=None, tool_calls=None))], created=1710775833, model='gemini/gemini-pro', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=67, prompt_tokens=10, total_tokens=77))
-------------
In realms of words, a dance takes place,
A symphony of rhythm, grace.
Each syllable a note so fine,
Weaving stories, making hearts entwine.

From whispers soft to thunder's roar,
Words paint worlds, forevermore.
They evoke emotions, deep and true,
Guiding us through life's every hue.
```

If you need any feedback or help, please visit the GitHub Issues page.