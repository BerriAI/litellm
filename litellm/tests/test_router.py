#### What this tests ####
# This tests litellm router

import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


def test_exception_raising():
    # this tests if the router raises an exception when invalid params are set
    # in this test both deployments have bad keys - Keep this test. It validates if the router raises the most recent exception
    litellm.set_verbose = True
    import openai

    try:
        print("testing if router raises an exception")
        old_api_key = os.environ["AZURE_API_KEY"]
        os.environ["AZURE_API_KEY"] = ""
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  #
                    "model": "gpt-3.5-turbo",
                    "api_key": "bad-key",
                },
                "tpm": 240000,
                "rpm": 1800,
            },
        ]
        router = Router(
            model_list=model_list,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=int(os.getenv("REDIS_PORT")),
            routing_strategy="simple-shuffle",
            set_verbose=False,
            num_retries=1,
        )  # type: ignore
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello this request will fail"}],
        )
        os.environ["AZURE_API_KEY"] = old_api_key
        pytest.fail(f"Should have raised an Auth Error")
    except openai.AuthenticationError:
        print(
            "Test Passed: Caught an OPENAI AUTH Error, Good job. This is what we needed!"
        )
        os.environ["AZURE_API_KEY"] = old_api_key
        router.reset()
    except Exception as e:
        os.environ["AZURE_API_KEY"] = old_api_key
        print("Got unexpected exception on router!", e)


# test_exception_raising()


def test_reading_key_from_model_list():
    # [PROD TEST CASE]
    # this tests if the router can read key from model list and make completion call, and completion + stream call. This is 90% of the router use case
    # DO NOT REMOVE THIS TEST. It's an IMP ONE. Speak to Ishaan, if you are tring to remove this
    litellm.set_verbose = False
    import openai

    try:
        print("testing if router raises an exception")
        old_api_key = os.environ["AZURE_API_KEY"]
        os.environ.pop("AZURE_API_KEY", None)
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": old_api_key,
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            }
        ]

        router = Router(
            model_list=model_list,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=int(os.getenv("REDIS_PORT")),
            routing_strategy="simple-shuffle",
            set_verbose=True,
            num_retries=1,
        )  # type: ignore
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello this request will fail"}],
        )
        print("\n response", response)
        str_response = response.choices[0].message.content
        print("\n str_response", str_response)
        assert len(str_response) > 0

        print("\n Testing streaming response")
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello this request will fail"}],
            stream=True,
        )
        completed_response = ""
        for chunk in response:
            if chunk is not None:
                print(chunk)
                completed_response += chunk.choices[0].delta.content or ""
        print("\n completed_response", completed_response)
        assert len(completed_response) > 0
        print("\n Passed Streaming")
        os.environ["AZURE_API_KEY"] = old_api_key
        router.reset()
    except Exception as e:
        os.environ["AZURE_API_KEY"] = old_api_key
        print(f"FAILED TEST")
        pytest.fail(f"Got unexpected exception on router! - {e}")


# test_reading_key_from_model_list()


def test_call_one_endpoint():
    # [PROD TEST CASE]
    # user passes one deployment they want to call on the router, we call the specified one
    # this test makes a completion calls azure/chatgpt-v-2, it should work
    try:
        print("Testing calling a specific deployment")
        old_api_key = os.environ["AZURE_API_KEY"]

        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": old_api_key,
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "azure/azure-embedding-model",
                    "api_key": os.environ["AZURE_API_KEY"],
                    "api_base": os.environ["AZURE_API_BASE"],
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]
        litellm.set_verbose = True
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            set_verbose=True,
            num_retries=1,
        )  # type: ignore
        old_api_base = os.environ.pop("AZURE_API_BASE", None)

        async def call_azure_completion():
            response = await router.acompletion(
                model="azure/chatgpt-v-2",
                messages=[{"role": "user", "content": "hello this request will pass"}],
                specific_deployment=True,
            )
            print("\n response", response)

        async def call_azure_embedding():
            response = await router.aembedding(
                model="azure/azure-embedding-model",
                input=["good morning from litellm"],
                specific_deployment=True,
            )

            print("\n response", response)

        asyncio.run(call_azure_completion())
        asyncio.run(call_azure_embedding())

        os.environ["AZURE_API_BASE"] = old_api_base
        os.environ["AZURE_API_KEY"] = old_api_key
    except Exception as e:
        print(f"FAILED TEST")
        pytest.fail(f"Got unexpected exception on router! - {e}")


# test_call_one_endpoint()


def test_router_azure_acompletion():
    # [PROD TEST CASE]
    # This is 90% of the router use case, makes an acompletion call, acompletion + stream call and verifies it got a response
    # DO NOT REMOVE THIS TEST. It's an IMP ONE. Speak to Ishaan, if you are tring to remove this
    litellm.set_verbose = False
    import openai

    try:
        print("Router Test Azure - Acompletion, Acompletion with stream")

        # remove api key from env to repro how proxy passes key to router
        old_api_key = os.environ["AZURE_API_KEY"]
        os.environ.pop("AZURE_API_KEY", None)

        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": old_api_key,
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-turbo",
                    "api_key": os.getenv("AZURE_FRANCE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": "https://openai-france-1234.openai.azure.com",
                },
                "rpm": 1800,
            },
        ]

        router = Router(
            model_list=model_list, routing_strategy="simple-shuffle", set_verbose=True
        )  # type: ignore

        async def test1():
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hello this request will pass"}],
            )
            str_response = response.choices[0].message.content
            print("\n str_response", str_response)
            assert len(str_response) > 0
            print("\n response", response)

        asyncio.run(test1())

        print("\n Testing streaming response")

        async def test2():
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hello this request will fail"}],
                stream=True,
            )
            completed_response = ""
            async for chunk in response:
                if chunk is not None:
                    print(chunk)
                    completed_response += chunk.choices[0].delta.content or ""
            print("\n completed_response", completed_response)
            assert len(completed_response) > 0

        asyncio.run(test2())
        print("\n Passed Streaming")
        os.environ["AZURE_API_KEY"] = old_api_key
        router.reset()
    except Exception as e:
        os.environ["AZURE_API_KEY"] = old_api_key
        print(f"FAILED TEST")
        pytest.fail(f"Got unexpected exception on router! - {e}")


# test_router_azure_acompletion()

### FUNCTION CALLING


def test_function_calling():
    model_list = [
        {
            "model_name": "gpt-3.5-turbo-0613",
            "litellm_params": {
                "model": "gpt-3.5-turbo-0613",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 100000,
            "rpm": 10000,
        },
    ]

    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    router = Router(model_list=model_list)
    response = router.completion(
        model="gpt-3.5-turbo-0613", messages=messages, functions=functions
    )
    router.reset()
    print(response)


# test_acompletion_on_router()


def test_function_calling_on_router():
    try:
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]
        function1 = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ]
        router = Router(
            model_list=model_list,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=os.getenv("REDIS_PORT"),
        )
        messages = [{"role": "user", "content": "what's the weather in boston"}]
        response = router.completion(
            model="gpt-3.5-turbo", messages=messages, functions=function1
        )
        print(f"final returned response: {response}")
        router.reset()
        assert isinstance(response["choices"][0]["message"]["function_call"], dict)
    except Exception as e:
        print(f"An exception occurred: {e}")


# test_function_calling_on_router()


### IMAGE GENERATION
@pytest.mark.asyncio
async def test_aimg_gen_on_router():
    litellm.set_verbose = True
    try:
        model_list = [
            {
                "model_name": "dall-e-3",
                "litellm_params": {
                    "model": "dall-e-3",
                },
            },
            {
                "model_name": "dall-e-3",
                "litellm_params": {
                    "model": "azure/dall-e-3-test",
                    "api_version": "2023-12-01-preview",
                    "api_base": os.getenv("AZURE_SWEDEN_API_BASE"),
                    "api_key": os.getenv("AZURE_SWEDEN_API_KEY"),
                },
            },
            {
                "model_name": "dall-e-2",
                "litellm_params": {
                    "model": "azure/",
                    "api_version": "2023-06-01-preview",
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_key": os.getenv("AZURE_API_KEY"),
                },
            },
        ]
        router = Router(model_list=model_list, num_retries=3)
        response = await router.aimage_generation(
            model="dall-e-3", prompt="A cute baby sea otter"
        )
        print(response)
        assert len(response.data) > 0

        response = await router.aimage_generation(
            model="dall-e-2", prompt="A cute baby sea otter"
        )
        print(response)
        assert len(response.data) > 0

        router.reset()
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        elif "Operation polling timed out" in str(e):
            pass
        else:
            traceback.print_exc()
            pytest.fail(f"Error occurred: {e}")


# asyncio.run(test_aimg_gen_on_router())


def test_img_gen_on_router():
    litellm.set_verbose = True
    try:
        model_list = [
            {
                "model_name": "dall-e-3",
                "litellm_params": {
                    "model": "dall-e-3",
                },
            },
            {
                "model_name": "dall-e-3",
                "litellm_params": {
                    "model": "azure/dall-e-3-test",
                    "api_version": "2023-12-01-preview",
                    "api_base": os.getenv("AZURE_SWEDEN_API_BASE"),
                    "api_key": os.getenv("AZURE_SWEDEN_API_KEY"),
                },
            },
        ]
        router = Router(model_list=model_list)
        response = router.image_generation(
            model="dall-e-3", prompt="A cute baby sea otter"
        )
        print(response)
        assert len(response.data) > 0
        router.reset()
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_img_gen_on_router()
###


def test_aembedding_on_router():
    litellm.set_verbose = True
    try:
        model_list = [
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]
        router = Router(model_list=model_list)

        async def embedding_call():
            response = await router.aembedding(
                model="text-embedding-ada-002",
                input=["good morning from litellm", "this is another item"],
            )
            print(response)

        asyncio.run(embedding_call())

        print("\n Making sync Embedding call\n")
        response = router.embedding(
            model="text-embedding-ada-002",
            input=["good morning from litellm 2"],
        )
        router.reset()
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        elif "Operation polling timed out" in str(e):
            pass
        else:
            traceback.print_exc()
            pytest.fail(f"Error occurred: {e}")


# test_aembedding_on_router()


def test_azure_embedding_on_router():
    """
    [PROD Use Case] - Makes an aembedding call + embedding call
    """
    litellm.set_verbose = True
    try:
        model_list = [
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "azure/azure-embedding-model",
                    "api_key": os.environ["AZURE_API_KEY"],
                    "api_base": os.environ["AZURE_API_BASE"],
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]
        router = Router(model_list=model_list)

        async def embedding_call():
            response = await router.aembedding(
                model="text-embedding-ada-002", input=["good morning from litellm"]
            )
            print(response)

        asyncio.run(embedding_call())

        print("\n Making sync Azure Embedding call\n")

        response = router.embedding(
            model="text-embedding-ada-002",
            input=["test 2 from litellm. async embedding"],
        )
        print(response)
        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_azure_embedding_on_router()


def test_bedrock_on_router():
    litellm.set_verbose = True
    print("\n Testing bedrock on router\n")
    try:
        model_list = [
            {
                "model_name": "claude-v1",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-instant-v1",
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]

        async def test():
            router = Router(model_list=model_list)
            response = await router.acompletion(
                model="claude-v1",
                messages=[
                    {
                        "role": "user",
                        "content": "hello from litellm test",
                    }
                ],
            )
            print(response)
            router.reset()

        asyncio.run(test())
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_bedrock_on_router()


# test openai-compatible endpoint
@pytest.mark.asyncio
async def test_mistral_on_router():
    litellm.set_verbose = True
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "mistral/mistral-medium",
            },
        },
    ]
    router = Router(model_list=model_list)
    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello from litellm test",
            }
        ],
    )
    print(response)


# asyncio.run(test_mistral_on_router())


def test_openai_completion_on_router():
    # [PROD Use Case] - Makes an acompletion call + async acompletion call, and sync acompletion call, sync completion + stream
    # 4 LLM API calls made here. If it fails, add retries. Do not remove this test.
    litellm.set_verbose = True
    print("\n Testing OpenAI on router\n")
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
        ]
        router = Router(model_list=model_list)

        async def test():
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "hello from litellm test",
                    }
                ],
            )
            print(response)
            assert len(response.choices[0].message.content) > 0

            print("\n streaming + acompletion test")
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"hello from litellm test {time.time()}",
                    }
                ],
                stream=True,
            )
            complete_response = ""
            print(response)
            # if you want to see all the attributes and methods
            async for chunk in response:
                print(chunk)
                complete_response += chunk.choices[0].delta.content or ""
            print("\n complete response: ", complete_response)
            assert len(complete_response) > 0

        asyncio.run(test())
        print("\n Testing Sync completion calls \n")
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "hello from litellm test2",
                }
            ],
        )
        print(response)
        assert len(response.choices[0].message.content) > 0

        print("\n streaming + completion test")
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "hello from litellm test3",
                }
            ],
            stream=True,
        )
        complete_response = ""
        print(response)
        for chunk in response:
            print(chunk)
            complete_response += chunk.choices[0].delta.content or ""
        print("\n complete response: ", complete_response)
        assert len(complete_response) > 0
        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_openai_completion_on_router()


def test_reading_keys_os_environ():
    import openai

    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "os.environ/AZURE_API_KEY",
                    "api_base": "os.environ/AZURE_API_BASE",
                    "api_version": "os.environ/AZURE_API_VERSION",
                    "timeout": "os.environ/AZURE_TIMEOUT",
                    "stream_timeout": "os.environ/AZURE_STREAM_TIMEOUT",
                    "max_retries": "os.environ/AZURE_MAX_RETRIES",
                },
            },
        ]

        router = Router(model_list=model_list)
        for model in router.model_list:
            assert (
                model["litellm_params"]["api_key"] == os.environ["AZURE_API_KEY"]
            ), f"{model['litellm_params']['api_key']} vs {os.environ['AZURE_API_KEY']}"
            assert (
                model["litellm_params"]["api_base"] == os.environ["AZURE_API_BASE"]
            ), f"{model['litellm_params']['api_base']} vs {os.environ['AZURE_API_BASE']}"
            assert (
                model["litellm_params"]["api_version"]
                == os.environ["AZURE_API_VERSION"]
            ), f"{model['litellm_params']['api_version']} vs {os.environ['AZURE_API_VERSION']}"
            assert float(model["litellm_params"]["timeout"]) == float(
                os.environ["AZURE_TIMEOUT"]
            ), f"{model['litellm_params']['timeout']} vs {os.environ['AZURE_TIMEOUT']}"
            assert float(model["litellm_params"]["stream_timeout"]) == float(
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{model['litellm_params']['stream_timeout']} vs {os.environ['AZURE_STREAM_TIMEOUT']}"
            assert int(model["litellm_params"]["max_retries"]) == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{model['litellm_params']['max_retries']} vs {os.environ['AZURE_MAX_RETRIES']}"
            print("passed testing of reading keys from os.environ")
            model_id = model["model_info"]["id"]
            async_client: openai.AsyncAzureOpenAI = router.cache.get_cache(f"{model_id}_async_client")  # type: ignore
            assert async_client.api_key == os.environ["AZURE_API_KEY"]
            assert async_client.base_url == os.environ["AZURE_API_BASE"]
            assert async_client.max_retries == (
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{async_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert async_client.timeout == (
                os.environ["AZURE_TIMEOUT"]
            ), f"{async_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("async client set correctly!")

            print("\n Testing async streaming client")

            stream_async_client: openai.AsyncAzureOpenAI = router.cache.get_cache(f"{model_id}_stream_async_client")  # type: ignore
            assert stream_async_client.api_key == os.environ["AZURE_API_KEY"]
            assert stream_async_client.base_url == os.environ["AZURE_API_BASE"]
            assert stream_async_client.max_retries == (
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{stream_async_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert stream_async_client.timeout == (
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{stream_async_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("async stream client set correctly!")

            print("\n Testing sync client")
            client: openai.AzureOpenAI = router.cache.get_cache(f"{model_id}_client")  # type: ignore
            assert client.api_key == os.environ["AZURE_API_KEY"]
            assert client.base_url == os.environ["AZURE_API_BASE"]
            assert client.max_retries == (
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert client.timeout == (
                os.environ["AZURE_TIMEOUT"]
            ), f"{client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("sync client set correctly!")

            print("\n Testing sync stream client")
            stream_client: openai.AzureOpenAI = router.cache.get_cache(f"{model_id}_stream_client")  # type: ignore
            assert stream_client.api_key == os.environ["AZURE_API_KEY"]
            assert stream_client.base_url == os.environ["AZURE_API_BASE"]
            assert stream_client.max_retries == (
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{stream_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert stream_client.timeout == (
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{stream_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("sync stream client set correctly!")

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_reading_keys_os_environ()


def test_reading_openai_keys_os_environ():
    import openai

    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "os.environ/OPENAI_API_KEY",
                    "timeout": "os.environ/AZURE_TIMEOUT",
                    "stream_timeout": "os.environ/AZURE_STREAM_TIMEOUT",
                    "max_retries": "os.environ/AZURE_MAX_RETRIES",
                },
            },
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "os.environ/OPENAI_API_KEY",
                    "timeout": "os.environ/AZURE_TIMEOUT",
                    "stream_timeout": "os.environ/AZURE_STREAM_TIMEOUT",
                    "max_retries": "os.environ/AZURE_MAX_RETRIES",
                },
            },
        ]

        router = Router(model_list=model_list)
        for model in router.model_list:
            assert (
                model["litellm_params"]["api_key"] == os.environ["OPENAI_API_KEY"]
            ), f"{model['litellm_params']['api_key']} vs {os.environ['AZURE_API_KEY']}"
            assert float(model["litellm_params"]["timeout"]) == float(
                os.environ["AZURE_TIMEOUT"]
            ), f"{model['litellm_params']['timeout']} vs {os.environ['AZURE_TIMEOUT']}"
            assert float(model["litellm_params"]["stream_timeout"]) == float(
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{model['litellm_params']['stream_timeout']} vs {os.environ['AZURE_STREAM_TIMEOUT']}"
            assert int(model["litellm_params"]["max_retries"]) == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{model['litellm_params']['max_retries']} vs {os.environ['AZURE_MAX_RETRIES']}"
            print("passed testing of reading keys from os.environ")
            model_id = model["model_info"]["id"]
            async_client: openai.AsyncOpenAI = router.cache.get_cache(key=f"{model_id}_async_client")  # type: ignore
            assert async_client.api_key == os.environ["OPENAI_API_KEY"]
            assert async_client.max_retries == (
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{async_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert async_client.timeout == (
                os.environ["AZURE_TIMEOUT"]
            ), f"{async_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("async client set correctly!")

            print("\n Testing async streaming client")

            stream_async_client: openai.AsyncOpenAI = router.cache.get_cache(key=f"{model_id}_stream_async_client")  # type: ignore
            assert stream_async_client.api_key == os.environ["OPENAI_API_KEY"]
            assert stream_async_client.max_retries == (
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{stream_async_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert stream_async_client.timeout == (
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{stream_async_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("async stream client set correctly!")

            print("\n Testing sync client")
            client: openai.AzureOpenAI = router.cache.get_cache(key=f"{model_id}_client")  # type: ignore
            assert client.api_key == os.environ["OPENAI_API_KEY"]
            assert client.max_retries == (
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert client.timeout == (
                os.environ["AZURE_TIMEOUT"]
            ), f"{client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("sync client set correctly!")

            print("\n Testing sync stream client")
            stream_client: openai.AzureOpenAI = router.cache.get_cache(key=f"{model_id}_stream_client")  # type: ignore
            assert stream_client.api_key == os.environ["OPENAI_API_KEY"]
            assert stream_client.max_retries == (
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{stream_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert stream_client.timeout == (
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{stream_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("sync stream client set correctly!")

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_reading_openai_keys_os_environ()


def test_router_anthropic_key_dynamic():
    anthropic_api_key = os.environ.pop("ANTHROPIC_API_KEY")
    model_list = [
        {
            "model_name": "anthropic-claude",
            "litellm_params": {
                "model": "claude-instant-1.2",
                "api_key": anthropic_api_key,
            },
        }
    ]

    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    router.completion(model="anthropic-claude", messages=messages)
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key


def test_router_timeout():
    litellm.set_verbose = True
    from litellm._logging import verbose_logger
    import logging

    verbose_logger.setLevel(logging.DEBUG)
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "os.environ/OPENAI_API_KEY",
            },
        }
    ]
    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    start_time = time.time()
    try:
        res = router.completion(
            model="gpt-3.5-turbo", messages=messages, timeout=0.0001
        )
        print(res)
        pytest.fail("this should have timed out")
    except litellm.exceptions.Timeout as e:
        print("got timeout exception")
        print(e)
        print(vars(e))
        pass


@pytest.mark.asyncio
async def test_router_amoderation():
    model_list = [
        {
            "model_name": "openai-moderations",
            "litellm_params": {
                "model": "text-moderation-stable",
                "api_key": os.getenv("OPENAI_API_KEY", None),
            },
        }
    ]

    router = Router(model_list=model_list)
    result = await router.amoderation(
        model="openai-moderations", input="this is valid good text"
    )

    print("moderation result", result)
