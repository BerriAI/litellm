# AI/ML API
https://docs.aimlapi.com/quickstart/setting-up

## Usage
You can choose from LLama, Qwen, Flux, and 200+ other open and closed-source models on aimlapi.com/models. For example:

```python
import litellm

response = litellm.completion(
    model="openai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", # must starts with openai/ prefix
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
    model="openai/gpt-4o",  # must starts with openai/ prefix
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
        model="openai/gpt-4o",  # must starts with openai/ prefix
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
            model="openai/gpt-4o", # must starts with openai/ prefix
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
        model="openai/text-embedding-3-small", # must starts with openai/ prefix
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
        model="openai/dall-e-3",  # must starts with openai/ prefix
        api_key="",  # your aiml api-key
        api_base="https://api.aimlapi.com/v1", # 👈 the URL has changed from v2 to v1
        prompt="A cute baby sea otter",
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```