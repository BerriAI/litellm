#### What this tests ####
#    This tests the timeout decorator

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
import litellm
import openai
import pytest, uuid


def test_timeout():
    # this Will Raise a timeout
    litellm.set_verbose = False
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            timeout=0.01,
            messages=[{"role": "user", "content": "hello, write a 20 pg essay"}],
        )
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


# test_timeout()


def test_bedrock_timeout():
    # this Will Raise a timeout
    litellm.set_verbose = True
    try:
        response = litellm.completion(
            model="bedrock/anthropic.claude-instant-v1",
            timeout=0.01,
            messages=[{"role": "user", "content": "hello, write a 20 pg essay"}],
        )
        pytest.fail("Did not raise error `openai.APITimeoutError`")
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


def test_hanging_request_azure():
    litellm.set_verbose = True
    import asyncio

    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "azure-gpt",
                    "litellm_params": {
                        "model": "azure/chatgpt-v-2",
                        "api_base": os.environ["AZURE_API_BASE"],
                        "api_key": os.environ["AZURE_API_KEY"],
                    },
                },
                {
                    "model_name": "openai-gpt",
                    "litellm_params": {"model": "gpt-3.5-turbo"},
                },
            ]
        )

        encoded = litellm.utils.encode(model="gpt-3.5-turbo", text="blue")[0]

        async def _test():
            response = await router.acompletion(
                model="azure-gpt",
                messages=[
                    {"role": "user", "content": f"what color is red {uuid.uuid4()}"}
                ],
                logit_bias={encoded: 100},
                timeout=0.01,
            )
            print(response)
            return response

        response = asyncio.run(_test())

        if response.choices[0].message.content is not None:
            pytest.fail("Got a response, expected a timeout")
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


# test_hanging_request_azure()


def test_hanging_request_openai():
    litellm.set_verbose = True
    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "azure-gpt",
                    "litellm_params": {
                        "model": "azure/chatgpt-v-2",
                        "api_base": os.environ["AZURE_API_BASE"],
                        "api_key": os.environ["AZURE_API_KEY"],
                    },
                },
                {
                    "model_name": "openai-gpt",
                    "litellm_params": {"model": "gpt-3.5-turbo"},
                },
            ]
        )

        encoded = litellm.utils.encode(model="gpt-3.5-turbo", text="blue")[0]
        response = router.completion(
            model="openai-gpt",
            messages=[{"role": "user", "content": "what color is red"}],
            logit_bias={encoded: 100},
            timeout=0.01,
        )
        print(response)

        if response.choices[0].message.content is not None:
            pytest.fail("Got a response, expected a timeout")
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


# test_hanging_request_openai()

# test_timeout()


def test_timeout_streaming():
    # this Will Raise a timeout
    litellm.set_verbose = False
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello, write a 20 pg essay"}],
            timeout=0.0001,
            stream=True,
        )
        for chunk in response:
            print(chunk)
    except openai.APITimeoutError as e:
        print(
            "Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e
        )
        print(type(e))
        pass
    except Exception as e:
        pytest.fail(
            f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}"
        )


# test_timeout_streaming()


def test_timeout_ollama():
    # this Will Raise a timeout
    import litellm

    litellm.set_verbose = True
    try:
        litellm.request_timeout = 0.1
        litellm.set_verbose = True
        response = litellm.completion(
            model="ollama/phi",
            messages=[{"role": "user", "content": "hello, what llm are u"}],
            max_tokens=1,
            api_base="https://test-ollama-endpoint.onrender.com",
        )
        # Add any assertions here to check the response
        litellm.request_timeout = None
        print(response)
    except openai.APITimeoutError as e:
        print("got a timeout error! Passed ! ")
        pass


# test_timeout_ollama()
