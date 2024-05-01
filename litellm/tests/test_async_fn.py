#### What this tests ####
#    This tests the the acompletion function #

import sys, os
import pytest
import traceback
import asyncio, logging

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, acompletion, acreate

litellm.num_retries = 3


def test_sync_response_anyscale():
    litellm.set_verbose = False
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = completion(
            model="anyscale/mistralai/Mistral-7B-Instruct-v0.1",
            messages=messages,
            timeout=5,
        )
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


# test_sync_response_anyscale()


def test_async_response_openai():
    import asyncio

    litellm.set_verbose = True

    async def test_get_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(
                model="gpt-3.5-turbo", messages=messages, timeout=5
            )
            print(f"response: {response}")
            print(f"response ms: {response._response_ms}")
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
            print(e)

    asyncio.run(test_get_response())


# test_async_response_openai()


def test_async_response_azure():
    import asyncio

    litellm.set_verbose = True

    async def test_get_response():
        user_message = "What do you know?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(
                model="azure/gpt-turbo",
                messages=messages,
                base_url=os.getenv("CLOUDFLARE_AZURE_BASE_URL"),
                api_key=os.getenv("AZURE_FRANCE_API_KEY"),
            )
            print(f"response: {response}")
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    asyncio.run(test_get_response())


# test_async_response_azure()


def test_async_anyscale_response():
    import asyncio

    litellm.set_verbose = True

    async def test_get_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(
                model="anyscale/mistralai/Mistral-7B-Instruct-v0.1",
                messages=messages,
                timeout=5,
            )
            # response = await response
            print(f"response: {response}")
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    asyncio.run(test_get_response())


# test_async_anyscale_response()


@pytest.mark.skip(reason="Flaky test-cloudflare is very unstable")
def test_async_completion_cloudflare():
    try:
        litellm.set_verbose = True

        async def test():
            response = await litellm.acompletion(
                model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                messages=[{"content": "what llm are you", "role": "user"}],
                max_tokens=5,
                num_retries=3,
            )
            print(response)
            return response

        response = asyncio.run(test())
        text_response = response["choices"][0]["message"]["content"]
        assert len(text_response) > 1  # more than 1 chars in response

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_async_completion_cloudflare()


@pytest.mark.skip(reason="Flaky test")
def test_get_cloudflare_response_streaming():
    import asyncio

    async def test_async_call():
        user_message = "write a short poem in one sentence"
        messages = [{"content": user_message, "role": "user"}]
        try:
            litellm.set_verbose = False
            response = await acompletion(
                model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                messages=messages,
                stream=True,
                num_retries=3,  # cloudflare ai workers is EXTREMELY UNSTABLE
            )
            print(type(response))

            import inspect

            is_async_generator = inspect.isasyncgen(response)
            print(is_async_generator)

            output = ""
            i = 0
            async for chunk in response:
                print(chunk)
                token = chunk["choices"][0]["delta"].get("content", "")
                if token == None:
                    continue  # openai v1.0.0 returns content=None
                output += token
            assert output is not None, "output cannot be None."
            assert isinstance(output, str), "output needs to be of type str"
            assert len(output) > 0, "Length of output needs to be greater than 0."
            print(f"output: {output}")
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    asyncio.run(test_async_call())


@pytest.mark.asyncio
async def test_hf_completion_tgi():
    # litellm.set_verbose=True
    try:
        response = await acompletion(
            model="huggingface/HuggingFaceH4/zephyr-7b-beta",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        # Add any assertions here to check the response
        print(response)
    except litellm.APIError as e:
        print("got an api error")
        pass
    except litellm.Timeout as e:
        print("got a timeout error")
        pass
    except litellm.RateLimitError as e:
        # this will catch the model is overloaded error
        print("got a rate limit error")
        pass
    except Exception as e:
        if "Model is overloaded" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred: {e}")


# test_get_cloudflare_response_streaming()


@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_completion_sagemaker():
    # litellm.set_verbose=True
    try:
        response = await acompletion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        # Add any assertions here to check the response
        print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_get_response_streaming():
    import asyncio

    async def test_async_call():
        user_message = "write a short poem in one sentence"
        messages = [{"content": user_message, "role": "user"}]
        try:
            litellm.set_verbose = True
            response = await acompletion(
                model="gpt-3.5-turbo", messages=messages, stream=True, timeout=5
            )
            print(type(response))

            import inspect

            is_async_generator = inspect.isasyncgen(response)
            print(is_async_generator)

            output = ""
            i = 0
            async for chunk in response:
                token = chunk["choices"][0]["delta"].get("content", "")
                if token == None:
                    continue  # openai v1.0.0 returns content=None
                output += token
            assert output is not None, "output cannot be None."
            assert isinstance(output, str), "output needs to be of type str"
            assert len(output) > 0, "Length of output needs to be greater than 0."
            print(f"output: {output}")
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    asyncio.run(test_async_call())


# test_get_response_streaming()


def test_get_response_non_openai_streaming():
    import asyncio

    litellm.set_verbose = True
    litellm.num_retries = 0

    async def test_async_call():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(
                model="anyscale/mistralai/Mistral-7B-Instruct-v0.1",
                messages=messages,
                stream=True,
                timeout=5,
            )
            print(type(response))

            import inspect

            is_async_generator = inspect.isasyncgen(response)
            print(is_async_generator)

            output = ""
            i = 0
            async for chunk in response:
                token = chunk["choices"][0]["delta"].get("content", None)
                if token == None:
                    continue
                print(token)
                output += token
            print(f"output: {output}")
            assert output is not None, "output cannot be None."
            assert isinstance(output, str), "output needs to be of type str"
            assert len(output) > 0, "Length of output needs to be greater than 0."
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        return response

    asyncio.run(test_async_call())


# test_get_response_non_openai_streaming()
