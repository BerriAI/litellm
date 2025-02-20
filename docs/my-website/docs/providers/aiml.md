# AI/ML API  

Getting started with the AI/ML API is simple. Follow these steps to set up your integration:

### 1. Get Your API Key  
To begin, you need an API key. You can obtain yours here:  
🔑 [Get Your API Key](https://aimlapi.com/app/keys/?utm_source=aimlapi&utm_medium=github&utm_campaign=integration)  

### 2. Explore Available Models  
Looking for a different model? Browse the full list of supported models:  
📚 [Full List of Models](https://docs.aimlapi.com/api-overview/model-database/text-models?utm_source=aimlapi&utm_medium=github&utm_campaign=integration)  

### 3. Read the Documentation  
For detailed setup instructions and usage guidelines, check out the official documentation:  
📖 [AI/ML API Docs](https://docs.aimlapi.com/quickstart/setting-up?utm_source=aimlapi&utm_medium=github&utm_campaign=integration)  

### 4. Need Help?  
If you have any questions, feel free to reach out. We’re happy to assist! 🚀  [Discord](https://discord.gg/hvaUsJpVJf)

## Usage
You can choose from LLama, Qwen, Flux, and 200+ other open and closed-source models on aimlapi.com/models. For example:

```python
import litellm

response = litellm.completion(
    model="openai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", # The model name must include prefix "openai" + the model name from ai/ml api
    api_key="", # your aiml api-key 
    api_base="https://api.aimlapi.com/v2",
    messages=[
        {
            "role": "user",
            "content": "Hey, how's it going?",
        }
    ],
)
```

## Streaming

```python
import litellm

response = litellm.completion(
    model="openai/Qwen/Qwen2-72B-Instruct",  # The model name must include prefix "openai" + the model name from ai/ml api
    api_key="",  # your aiml api-key 
    api_base="https://api.aimlapi.com/v2",
    messages=[
        {
            "role": "user",
            "content": "Hey, how's it going?",
        }
    ],
    stream=True,
)
for chunk in response:
    print(chunk)
```

## Async Completion

```python
import asyncio

import litellm


async def main():
    response = await litellm.acompletion(
        model="openai/anthropic/claude-3-5-haiku",  # The model name must include prefix "openai" + the model name from ai/ml api
        api_key="",  # your aiml api-key
        api_base="https://api.aimlapi.com/v2",
        messages=[
            {
                "role": "user",
                "content": "Hey, how's it going?",
            }
        ],
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

## Async Streaming

```python
import asyncio
import traceback

import litellm


async def main():
    try:
        print("test acompletion + streaming")
        response = await litellm.acompletion(
            model="openai/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", # The model name must include prefix "openai" + the model name from ai/ml api
            api_key="", # your aiml api-key
            api_base="https://api.aimlapi.com/v2",
            messages=[{"content": "Hey, how's it going?", "role": "user"}],
            stream=True,
        )
        print(f"response: {response}")
        async for chunk in response:
            print(chunk)
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


if __name__ == "__main__":
    asyncio.run(main())
```

## Async Embedding

```python
import asyncio

import litellm


async def main():
    response = await litellm.aembedding(
        model="openai/text-embedding-3-small", # The model name must include prefix "openai" + the model name from ai/ml api
        api_key="",  # your aiml api-key
        api_base="https://api.aimlapi.com/v1", # 👈 the URL has changed from v2 to v1
        input="Your text string",
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

## Async Image Generation

```python
import asyncio

import litellm


async def main():
    response = await litellm.aimage_generation(
        model="openai/dall-e-3",  # The model name must include prefix "openai" + the model name from ai/ml api
        api_key="",  # your aiml api-key
        api_base="https://api.aimlapi.com/v1", # 👈 the URL has changed from v2 to v1
        prompt="A cute baby sea otter",
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```