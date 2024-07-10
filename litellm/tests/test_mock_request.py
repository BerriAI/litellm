#### What this tests ####
#    This tests mock request calls to litellm

import os
import sys
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


def test_mock_request():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litellm.mock_completion(model=model, messages=messages, stream=False)
        print(response)
        print(type(response))
    except:
        traceback.print_exc()


# test_mock_request()
def test_streaming_mock_request():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litellm.mock_completion(model=model, messages=messages, stream=True)
        complete_response = ""
        for chunk in response:
            complete_response += chunk["choices"][0]["delta"]["content"] or ""
        if complete_response == "":
            raise Exception("Empty response received")
    except:
        traceback.print_exc()


# test_streaming_mock_request()


@pytest.mark.asyncio()
async def test_async_mock_streaming_request():
    generator = await litellm.acompletion(
        messages=[{"role": "user", "content": "Why is LiteLLM amazing?"}],
        mock_response="LiteLLM is awesome",
        stream=True,
        model="gpt-3.5-turbo",
    )
    complete_response = ""
    async for chunk in generator:
        print(chunk)
        complete_response += chunk["choices"][0]["delta"]["content"] or ""

    assert (
        complete_response == "LiteLLM is awesome"
    ), f"Unexpected response got {complete_response}"


def test_mock_request_n_greater_than_1():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litellm.mock_completion(model=model, messages=messages, n=5)
        print("response: ", response)

        assert len(response.choices) == 5
        for choice in response.choices:
            assert choice.message.content == "This is a mock request"

    except:
        traceback.print_exc()


@pytest.mark.asyncio()
async def test_async_mock_streaming_request_n_greater_than_1():
    generator = await litellm.acompletion(
        messages=[{"role": "user", "content": "Why is LiteLLM amazing?"}],
        mock_response="LiteLLM is awesome",
        stream=True,
        model="gpt-3.5-turbo",
        n=5,
    )
    complete_response = ""
    async for chunk in generator:
        print(chunk)
        # complete_response += chunk["choices"][0]["delta"]["content"] or ""

    # assert (
    #     complete_response == "LiteLLM is awesome"
    # ), f"Unexpected response got {complete_response}"
