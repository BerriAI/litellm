import asyncio
import litellm

# litellm.log_level = "verbose"

base_model = "google/gemma-3-4b-it"

model = f"bytez/{base_model}"


kwargs = {
    "max_tokens": 50,
    #
    "repetition_penalty": 1.2,
}

results = litellm.get_supported_openai_params(model=model, custom_llm_provider="bytez")

a = 2


def test_sync_completion(messages):
    response = litellm.completion(
        model=model, messages=messages, **kwargs  # type: ignore
    )

    print("Response is: ", response.choices[0].message.content)
    pass


def test_sync_streaming(messages):
    response = litellm.completion(
        model=model, messages=messages, stream=True, **kwargs  # type: ignore
    )

    for chunk in response:
        # last chunk is None
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content)


# async
async def test_async_completion(messages):
    response = await litellm.acompletion(
        model=model, messages=messages, **kwargs  # type: ignore
    )

    print("Response is: ", response.choices[0].message.content)


async def test_async_streaming(messages):
    response = await litellm.acompletion(
        model=model, messages=messages, stream=True, **kwargs  # type: ignore
    )

    # Iterate through the streaming response
    # NOTE this will have the finish_reason: "stop" set implicitly
    # the streaming handler in streaming_handler.py will need to be updated in the future in order to support
    # stopping streams mid run
    async for chunk in response:  # type: ignore

        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content)

    pass


messages_list = [
    [
        {
            "role": "user",
            "content": "What color is this cat?",
        }
    ],
    [
        {
            "role": "user",
            "content": {"type": "text", "text": "What color is this cat?"},
        }
    ],
    [
        {
            "role": "user",
            "content": [
                "What color is this cat?",
                {
                    "type": "image_url",
                    "url": "https://images.squarespace-cdn.com/content/v1/5452d441e4b0c188b51fef1a/1615326541809-TW01PVTOJ4PXQUXVRLHI/male-orange-tabby-cat.jpg",
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this cat?"},
                {
                    "type": "image_url",
                    "url": "https://images.squarespace-cdn.com/content/v1/5452d441e4b0c188b51fef1a/1615326541809-TW01PVTOJ4PXQUXVRLHI/male-orange-tabby-cat.jpg",
                },
            ],
        }
    ],
]


for index, messages in enumerate(messages_list, 0):
    print(f"On index: {index} of message tests")

    test_sync_completion(messages)
    test_sync_streaming(messages)
    asyncio.run(test_async_completion(messages))
    asyncio.run(test_async_streaming(messages))

    a = 2
