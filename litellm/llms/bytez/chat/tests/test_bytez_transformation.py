import asyncio
import litellm

from messages_list import messages_list

base_model = "google/gemma-3-4b-it"

model = f"bytez/{base_model}"


kwargs = {
    "max_tokens": 50,
    #
    "repetition_penalty": 1.2,
}

results = litellm.get_supported_openai_params(model=model, custom_llm_provider="bytez")


def test_sync_completion(messages):
    response = litellm.completion(
        model=model, messages=messages, **kwargs  # type: ignore
    )

    content = response.choices[0].message.content

    print("Response is: ", content)

    assert isinstance(content, str) and len(content) > 0


def test_sync_streaming(messages):
    response = litellm.completion(
        model=model, messages=messages, stream=True, **kwargs  # type: ignore
    )

    content_chunks = []

    for chunk in response:
        # last chunk is None
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content)
            content_chunks.append(content)

    assert len(content_chunks) > 0


# async
async def test_async_completion(messages):
    response = await litellm.acompletion(
        model=model, messages=messages, **kwargs  # type: ignore
    )

    content = response.choices[0].message.content

    print("Response is: ", content)

    assert isinstance(content, str) and len(content) > 0


async def test_async_streaming(messages):
    response = await litellm.acompletion(
        model=model, messages=messages, stream=True, **kwargs  # type: ignore
    )

    # Iterate through the streaming response
    # NOTE this will have the finish_reason: "stop" set implicitly
    # the streaming handler in streaming_handler.py will need to be updated in the future in order to support
    # stopping streams mid run

    content_chunks = []
    async for chunk in response:  # type: ignore

        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content)
            content_chunks.append(content)

    assert len(content_chunks) > 0


for index, messages in enumerate(messages_list, 0):
    print(f"On index: {index} of message tests")

    test_sync_completion(messages)
    test_sync_streaming(messages)
    asyncio.run(test_async_completion(messages))
    asyncio.run(test_async_streaming(messages))
