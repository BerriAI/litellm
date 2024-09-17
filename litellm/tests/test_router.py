#### What this tests ####
# This tests litellm router

import asyncio
import os
import sys
import time
import traceback

import openai
import pytest

import litellm.types
import litellm.types.router

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.router_utils.cooldown_handlers import (
    _async_get_cooldown_deployments,
    _get_cooldown_deployments,
)
from litellm.types.router import DeploymentTypedDict

load_dotenv()


def test_router_deployment_typing():
    deployment_typed_dict = DeploymentTypedDict(
        model_name="hi", litellm_params={"model": "hello-world"}
    )
    for value in deployment_typed_dict.items():
        assert not isinstance(value, BaseModel)


def test_router_multi_org_list():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "my-key",
                    "api_base": "https://api.openai.com/v1",
                    "organization": ["org-1", "org-2", "org-3"],
                },
            }
        ]
    )

    assert len(router.get_model_list()) == 3


@pytest.mark.asyncio()
async def test_router_provider_wildcard_routing():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "api_base": "https://api.openai.com/v1",
                },
            },
            {
                "model_name": "anthropic/*",
                "litellm_params": {
                    "model": "anthropic/*",
                    "api_key": os.environ["ANTHROPIC_API_KEY"],
                },
            },
            {
                "model_name": "groq/*",
                "litellm_params": {
                    "model": "groq/*",
                    "api_key": os.environ["GROQ_API_KEY"],
                },
            },
        ]
    )

    print("router model list = ", router.get_model_list())

    response1 = await router.acompletion(
        model="anthropic/claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 1 = ", response1)

    response2 = await router.acompletion(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 2 = ", response2)

    response3 = await router.acompletion(
        model="groq/llama3-8b-8192",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 3 = ", response3)


def test_router_specific_model_via_id():
    """
    Call a specific deployment by it's id
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-fake-key",
                    "mock_response": "Hello world",
                },
                "model_info": {"id": "1234"},
            }
        ]
    )

    router.completion(model="1234", messages=[{"role": "user", "content": "Hey!"}])


def test_router_azure_ai_client_init():

    _deployment = {
        "model_name": "meta-llama-3-70b",
        "litellm_params": {
            "model": "azure_ai/Meta-Llama-3-70B-instruct",
            "api_base": "my-fake-route",
            "api_key": "my-fake-key",
        },
        "model_info": {"id": "1234"},
    }
    router = Router(model_list=[_deployment])

    _client = router._get_client(
        deployment=_deployment,
        client_type="async",
        kwargs={"stream": False},
    )
    print(_client)
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    assert isinstance(_client, AsyncOpenAI)
    assert not isinstance(_client, AsyncAzureOpenAI)


def test_router_sensitive_keys():
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",  # openai model name
                    "litellm_params": {  # params for litellm completion/embedding call
                        "model": "azure/chatgpt-v-2",
                        "api_key": "special-key",
                    },
                    "model_info": {"id": 12345},
                },
            ],
        )
    except Exception as e:
        print(f"error msg - {str(e)}")
        assert "special-key" not in str(e)


def test_router_order():
    """
    Asserts for 2 models in a model group, model with order=1 always called first
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "mock_response": "Hello world",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad-key",
                    "mock_response": Exception("this is a bad key"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
        enable_pre_call_checks=True,
    )

    for _ in range(100):
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )

        assert isinstance(response, litellm.ModelResponse)
        assert response._hidden_params["model_id"] == "1"


@pytest.mark.parametrize("num_retries", [None, 2])
@pytest.mark.parametrize("max_retries", [None, 4])
def test_router_num_retries_init(num_retries, max_retries):
    """
    - test when num_retries set v/s not
    - test client value when max retries set v/s not
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "max_retries": max_retries,
                },
                "model_info": {"id": 12345},
            },
        ],
        num_retries=num_retries,
    )

    if num_retries is not None:
        assert router.num_retries == num_retries
    else:
        assert router.num_retries == openai.DEFAULT_MAX_RETRIES

    model_client = router._get_client(
        {"model_info": {"id": 12345}}, client_type="async", kwargs={}
    )

    if max_retries is not None:
        assert getattr(model_client, "max_retries") == max_retries
    else:
        assert getattr(model_client, "max_retries") == 0


@pytest.mark.parametrize(
    "timeout", [10, 1.0, httpx.Timeout(timeout=300.0, connect=20.0)]
)
@pytest.mark.parametrize("ssl_verify", [True, False])
def test_router_timeout_init(timeout, ssl_verify):
    """
    Allow user to pass httpx.Timeout

    related issue - https://github.com/BerriAI/litellm/issues/3162
    """
    litellm.ssl_verify = ssl_verify

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "timeout": timeout,
                },
                "model_info": {"id": 1234},
            }
        ]
    )

    model_client = router._get_client(
        deployment={"model_info": {"id": 1234}}, client_type="sync_client", kwargs={}
    )

    assert getattr(model_client, "timeout") == timeout

    print(f"vars model_client: {vars(model_client)}")
    http_client = getattr(model_client, "_client")
    print(f"http client: {vars(http_client)}, ssl_Verify={ssl_verify}")
    if ssl_verify == False:
        assert http_client._transport._pool._ssl_context.verify_mode.name == "CERT_NONE"
    else:
        assert (
            http_client._transport._pool._ssl_context.verify_mode.name
            == "CERT_REQUIRED"
        )


@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_router_retries(sync_mode):
    """
    - make sure retries work as expected
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "bad-key"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-2",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": os.getenv("AZURE_API_VERSION"),
            },
        },
    ]

    router = Router(model_list=model_list, num_retries=2)

    if sync_mode:
        router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    else:
        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )

        print(response.choices[0].message)


@pytest.mark.parametrize(
    "mistral_api_base",
    [
        "os.environ/AZURE_MISTRAL_API_BASE",
        "https://Mistral-large-nmefg-serverless.eastus2.inference.ai.azure.com/v1/",
        "https://Mistral-large-nmefg-serverless.eastus2.inference.ai.azure.com/v1",
        "https://Mistral-large-nmefg-serverless.eastus2.inference.ai.azure.com/",
        "https://Mistral-large-nmefg-serverless.eastus2.inference.ai.azure.com",
    ],
)
def test_router_azure_ai_studio_init(mistral_api_base):
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "azure/mistral-large-latest",
                    "api_key": "os.environ/AZURE_MISTRAL_API_KEY",
                    "api_base": mistral_api_base,
                },
                "model_info": {"id": 1234},
            }
        ]
    )

    model_client = router._get_client(
        deployment={"model_info": {"id": 1234}}, client_type="sync_client", kwargs={}
    )
    url = getattr(model_client, "_base_url")
    uri_reference = str(getattr(url, "_uri_reference"))

    print(f"uri_reference: {uri_reference}")

    assert "/v1/" in uri_reference
    assert uri_reference.count("v1") == 1


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


def test_router_context_window_fallback():
    """
    - Give a gpt-3.5-turbo model group with different context windows (4k vs. 16k)
    - Send a 5k prompt
    - Assert it works
    """
    import os

    from large_text import text

    litellm.set_verbose = False

    print(f"len(text): {len(text)}")
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "base_model": "azure/gpt-35-turbo",
                },
            },
            {
                "model_name": "gpt-3.5-turbo-large",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-1106",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, context_window_fallbacks=[{"gpt-3.5-turbo": ["gpt-3.5-turbo-large"]}], num_retries=0)  # type: ignore

        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": text},
                {"role": "user", "content": "Who was Alexander?"},
            ],
        )

        print(f"response: {response}")
        assert response.model == "gpt-3.5-turbo-1106"
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")


@pytest.mark.asyncio
async def test_async_router_context_window_fallback():
    """
    - Give a gpt-3.5-turbo model group with different context windows (4k vs. 16k)
    - Send a 5k prompt
    - Assert it works
    """
    import os

    from large_text import text

    litellm.set_verbose = False

    print(f"len(text): {len(text)}")
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "base_model": "azure/gpt-35-turbo",
                },
            },
            {
                "model_name": "gpt-3.5-turbo-large",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-1106",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, context_window_fallbacks=[{"gpt-3.5-turbo": ["gpt-3.5-turbo-large"]}], num_retries=0)  # type: ignore

        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": text},
                {"role": "user", "content": "Who was Alexander?"},
            ],
        )

        print(f"response: {response}")
        assert response.model == "gpt-3.5-turbo-1106"
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")


def test_router_rpm_pre_call_check():
    """
    - for a given model not in model cost map
    - with rpm set
    - check if rpm check is run
    """
    try:
        model_list = [
            {
                "model_name": "fake-openai-endpoint",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "openai/my-fake-model",
                    "api_key": "my-fake-key",
                    "api_base": "https://openai-function-calling-workers.tasslexyz.workers.dev/",
                    "rpm": 0,
                },
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, enable_pre_call_checks=True, num_retries=0)  # type: ignore

        try:
            router._pre_call_checks(
                model="fake-openai-endpoint",
                healthy_deployments=model_list,
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )
            pytest.fail("Expected this to fail")
        except:
            pass
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")


def test_router_context_window_check_pre_call_check_in_group_custom_model_info():
    """
    - Give a gpt-3.5-turbo model group with different context windows (4k vs. 16k)
    - Send a 5k prompt
    - Assert it works
    """
    import os

    from large_text import text

    litellm.set_verbose = False

    print(f"len(text): {len(text)}")
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "base_model": "azure/gpt-35-turbo",
                    "mock_response": "Hello world 1!",
                },
                "model_info": {"max_input_tokens": 100},
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-1106",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "mock_response": "Hello world 2!",
                },
                "model_info": {"max_input_tokens": 0},
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, enable_pre_call_checks=True, num_retries=0)  # type: ignore

        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Who was Alexander?"},
            ],
        )

        print(f"response: {response}")

        assert response.choices[0].message.content == "Hello world 1!"
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")


def test_router_context_window_check_pre_call_check():
    """
    - Give a gpt-3.5-turbo model group with different context windows (4k vs. 16k)
    - Send a 5k prompt
    - Assert it works
    """
    import os

    from large_text import text

    litellm.set_verbose = False

    print(f"len(text): {len(text)}")
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "base_model": "azure/gpt-35-turbo",
                    "mock_response": "Hello world 1!",
                },
                "model_info": {"base_model": "azure/gpt-35-turbo"},
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-1106",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "mock_response": "Hello world 2!",
                },
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, enable_pre_call_checks=True, num_retries=0)  # type: ignore

        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": text},
                {"role": "user", "content": "Who was Alexander?"},
            ],
        )

        print(f"response: {response}")

        assert response.choices[0].message.content == "Hello world 2!"
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")


def test_router_context_window_check_pre_call_check_out_group():
    """
    - Give 2 gpt-3.5-turbo model groups with different context windows (4k vs. 16k)
    - Send a 5k prompt
    - Assert it works
    """
    import os

    from large_text import text

    litellm.set_verbose = False

    print(f"len(text): {len(text)}")
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo-small",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "base_model": "azure/gpt-35-turbo",
                },
            },
            {
                "model_name": "gpt-3.5-turbo-large",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-1106",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, enable_pre_call_checks=True, num_retries=0, context_window_fallbacks=[{"gpt-3.5-turbo-small": ["gpt-3.5-turbo-large"]}])  # type: ignore

        response = router.completion(
            model="gpt-3.5-turbo-small",
            messages=[
                {"role": "system", "content": text},
                {"role": "user", "content": "Who was Alexander?"},
            ],
        )

        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")


def test_filter_invalid_params_pre_call_check():
    """
    - gpt-3.5-turbo supports 'response_object'
    - gpt-3.5-turbo-16k doesn't support 'response_object'

    run pre-call check -> assert returned list doesn't include gpt-3.5-turbo-16k
    """
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-16k",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, enable_pre_call_checks=True, num_retries=0)  # type: ignore

        filtered_deployments = router._pre_call_checks(
            model="gpt-3.5-turbo",
            healthy_deployments=model_list,
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            request_kwargs={"response_format": {"type": "json_object"}},
        )
        assert len(filtered_deployments) == 1
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")


@pytest.mark.parametrize("allowed_model_region", ["eu", None])
def test_router_region_pre_call_check(allowed_model_region):
    """
    If region based routing set
    - check if only model in allowed region is allowed by '_pre_call_checks'
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/chatgpt-v-2",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "base_model": "azure/gpt-35-turbo",
                "region_name": "eu",
            },
            "model_info": {"id": "1"},
        },
        {
            "model_name": "gpt-3.5-turbo-large",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "gpt-3.5-turbo-1106",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "2"},
        },
    ]

    router = Router(model_list=model_list, enable_pre_call_checks=True)

    _healthy_deployments = router._pre_call_checks(
        model="gpt-3.5-turbo",
        healthy_deployments=model_list,
        messages=[{"role": "user", "content": "Hey!"}],
        request_kwargs={"allowed_model_region": allowed_model_region},
    )

    if allowed_model_region is None:
        assert len(_healthy_deployments) == 2
    else:
        assert len(_healthy_deployments) == 1, "No models selected as healthy"
        assert (
            _healthy_deployments[0]["model_info"]["id"] == "1"
        ), "Incorrect model id picked. Got id={}, expected id=1".format(
            _healthy_deployments[0]["model_info"]["id"]
        )


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
    except litellm.InternalServerError as e:
        pass
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        elif "Operation polling timed out" in str(e):
            pass
        elif "Connection error" in str(e):
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
        elif "Connection error" in str(e):
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


def test_model_group_info():
    router = Router(
        model_list=[
            {
                "model_name": "command-r-plus",
                "litellm_params": {"model": "cohere.command-r-plus-v1:0"},
            }
        ]
    )

    response = router.get_model_group_info(model_group="command-r-plus")

    assert response is not None


def test_consistent_model_id():
    """
    - For a given model group + litellm params, assert the model id is always the same

    Test on `_generate_model_id`

    Test on `set_model_list`

    Test on `_add_deployment`
    """
    model_group = "gpt-3.5-turbo"
    litellm_params = {
        "model": "openai/my-fake-model",
        "api_key": "my-fake-key",
        "api_base": "https://openai-function-calling-workers.tasslexyz.workers.dev/",
        "stream_timeout": 0.001,
    }

    id1 = Router()._generate_model_id(
        model_group=model_group, litellm_params=litellm_params
    )

    id2 = Router()._generate_model_id(
        model_group=model_group, litellm_params=litellm_params
    )

    assert id1 == id2


@pytest.mark.skip(reason="local test")
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
            assert async_client.max_retries == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{async_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert async_client.timeout == int(
                os.environ["AZURE_TIMEOUT"]
            ), f"{async_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("async client set correctly!")

            print("\n Testing async streaming client")

            stream_async_client: openai.AsyncAzureOpenAI = router.cache.get_cache(f"{model_id}_stream_async_client")  # type: ignore
            assert stream_async_client.api_key == os.environ["AZURE_API_KEY"]
            assert stream_async_client.base_url == os.environ["AZURE_API_BASE"]
            assert stream_async_client.max_retries == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{stream_async_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert stream_async_client.timeout == int(
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{stream_async_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("async stream client set correctly!")

            print("\n Testing sync client")
            client: openai.AzureOpenAI = router.cache.get_cache(f"{model_id}_client")  # type: ignore
            assert client.api_key == os.environ["AZURE_API_KEY"]
            assert client.base_url == os.environ["AZURE_API_BASE"]
            assert client.max_retries == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert client.timeout == int(
                os.environ["AZURE_TIMEOUT"]
            ), f"{client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("sync client set correctly!")

            print("\n Testing sync stream client")
            stream_client: openai.AzureOpenAI = router.cache.get_cache(f"{model_id}_stream_client")  # type: ignore
            assert stream_client.api_key == os.environ["AZURE_API_KEY"]
            assert stream_client.base_url == os.environ["AZURE_API_BASE"]
            assert stream_client.max_retries == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{stream_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert stream_client.timeout == int(
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{stream_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("sync stream client set correctly!")

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_reading_keys_os_environ()


@pytest.mark.skip(reason="local test")
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
            assert async_client.max_retries == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{async_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert async_client.timeout == int(
                os.environ["AZURE_TIMEOUT"]
            ), f"{async_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("async client set correctly!")

            print("\n Testing async streaming client")

            stream_async_client: openai.AsyncOpenAI = router.cache.get_cache(key=f"{model_id}_stream_async_client")  # type: ignore
            assert stream_async_client.api_key == os.environ["OPENAI_API_KEY"]
            assert stream_async_client.max_retries == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{stream_async_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert stream_async_client.timeout == int(
                os.environ["AZURE_STREAM_TIMEOUT"]
            ), f"{stream_async_client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("async stream client set correctly!")

            print("\n Testing sync client")
            client: openai.AzureOpenAI = router.cache.get_cache(key=f"{model_id}_client")  # type: ignore
            assert client.api_key == os.environ["OPENAI_API_KEY"]
            assert client.max_retries == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert client.timeout == int(
                os.environ["AZURE_TIMEOUT"]
            ), f"{client.timeout} vs {os.environ['AZURE_TIMEOUT']}"
            print("sync client set correctly!")

            print("\n Testing sync stream client")
            stream_client: openai.AzureOpenAI = router.cache.get_cache(key=f"{model_id}_stream_client")  # type: ignore
            assert stream_client.api_key == os.environ["OPENAI_API_KEY"]
            assert stream_client.max_retries == int(
                os.environ["AZURE_MAX_RETRIES"]
            ), f"{stream_client.max_retries} vs {os.environ['AZURE_MAX_RETRIES']}"
            assert stream_client.timeout == int(
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
    import logging

    from litellm._logging import verbose_logger

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


def test_router_add_deployment():
    initial_model_list = [
        {
            "model_name": "fake-openai-endpoint",
            "litellm_params": {
                "model": "openai/my-fake-model",
                "api_key": "my-fake-key",
                "api_base": "https://openai-function-calling-workers.tasslexyz.workers.dev/",
            },
        },
    ]
    router = Router(model_list=initial_model_list)

    init_model_id_list = router.get_model_ids()

    print(f"init_model_id_list: {init_model_id_list}")

    router.add_deployment(
        deployment=Deployment(
            model_name="gpt-instruct",
            litellm_params=LiteLLM_Params(model="gpt-3.5-turbo-instruct"),
            model_info=ModelInfo(),
        )
    )

    new_model_id_list = router.get_model_ids()

    print(f"new_model_id_list: {new_model_id_list}")

    assert len(new_model_id_list) > len(init_model_id_list)

    assert new_model_id_list[1] != new_model_id_list[0]


@pytest.mark.asyncio
async def test_router_text_completion_client():
    # This tests if we re-use the Async OpenAI client
    # This test fails when we create a new Async OpenAI client per request
    try:
        model_list = [
            {
                "model_name": "fake-openai-endpoint",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": os.getenv("OPENAI_API_KEY", None),
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
            }
        ]
        router = Router(model_list=model_list, debug_level="DEBUG", set_verbose=True)
        tasks = []
        for _ in range(300):
            tasks.append(
                router.atext_completion(
                    model="fake-openai-endpoint",
                    prompt="hello from litellm test",
                )
            )

        # Execute all coroutines concurrently
        responses = await asyncio.gather(*tasks)
        print(responses)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.fixture
def mock_response() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        **{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1699896916,
            "model": "gpt-3.5-turbo-0125",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{\n"location": "Boston, MA"\n}',
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )


@pytest.mark.asyncio
async def test_router_model_usage(mock_response):
    """
    Test if tracking used model tpm works as expected
    """
    model = "my-fake-model"
    model_tpm = 100
    setattr(
        mock_response,
        "usage",
        litellm.Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )

    print(f"mock_response: {mock_response}")
    model_tpm = 100
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "tpm": model_tpm,
                    "mock_response": mock_response,
                },
            }
        ]
    )

    allowed_fails = 1  # allow for changing b/w minutes

    for _ in range(2):
        try:
            _ = await llm_router.acompletion(
                model=model, messages=[{"role": "user", "content": "Hey!"}]
            )
            await asyncio.sleep(3)

            initial_usage_tuple = await llm_router.get_model_group_usage(
                model_group=model
            )
            initial_usage = initial_usage_tuple[0]

            # completion call - 10 tokens
            _ = await llm_router.acompletion(
                model=model, messages=[{"role": "user", "content": "Hey!"}]
            )

            await asyncio.sleep(3)
            updated_usage_tuple = await llm_router.get_model_group_usage(
                model_group=model
            )
            updated_usage = updated_usage_tuple[0]

            assert updated_usage == initial_usage + 10  # type: ignore
            break
        except Exception as e:
            if allowed_fails > 0:
                print(
                    f"Decrementing allowed_fails: {allowed_fails}.\nReceived error - {str(e)}"
                )
                allowed_fails -= 1
            else:
                print(f"allowed_fails: {allowed_fails}")
                raise e


@pytest.mark.skip(reason="Check if this is causing ci/cd issues.")
@pytest.mark.asyncio
async def test_is_proxy_set():
    """
    Assert if proxy is set
    """
    from httpx import AsyncHTTPTransport

    os.environ["HTTPS_PROXY"] = "https://proxy.example.com:8080"
    from openai import AsyncAzureOpenAI

    # Function to check if a proxy is set on the client
    # Function to check if a proxy is set on the client
    def check_proxy(client: httpx.AsyncClient) -> bool:
        print(f"client._mounts: {client._mounts}")
        assert len(client._mounts) == 1
        for k, v in client._mounts.items():
            assert isinstance(v, AsyncHTTPTransport)
        return True

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "azure/gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "mock_response": "hello world",
                },
                "model_info": {"id": "1"},
            }
        ]
    )

    _deployment = llm_router.get_deployment(model_id="1")
    model_client: AsyncAzureOpenAI = llm_router._get_client(
        deployment=_deployment, kwargs={}, client_type="async"
    )  # type: ignore

    assert check_proxy(client=model_client._client)


@pytest.mark.parametrize(
    "model, base_model, llm_provider",
    [
        ("azure/gpt-4", None, "azure"),
        ("azure/gpt-4", "azure/gpt-4-0125-preview", "azure"),
        ("gpt-4", None, "openai"),
    ],
)
def test_router_get_model_info(model, base_model, llm_provider):
    """
    Test if router get model info works based on provider

    For azure -> only if base model set
    For openai -> use model=
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": model,
                    "api_key": "my-fake-key",
                    "api_base": "my-fake-base",
                },
                "model_info": {"base_model": base_model, "id": "1"},
            }
        ]
    )

    deployment = router.get_deployment(model_id="1")

    assert deployment is not None

    if llm_provider == "openai" or (base_model is not None and llm_provider == "azure"):
        router.get_router_model_info(deployment=deployment.to_json())
    else:
        try:
            router.get_router_model_info(deployment=deployment.to_json())
            pytest.fail("Expected this to raise model not mapped error")
        except Exception as e:
            if "This model isn't mapped yet" in str(e):
                pass


@pytest.mark.parametrize(
    "model, base_model, llm_provider",
    [
        ("azure/gpt-4", None, "azure"),
        ("azure/gpt-4", "azure/gpt-4-0125-preview", "azure"),
        ("gpt-4", None, "openai"),
    ],
)
def test_router_context_window_pre_call_check(model, base_model, llm_provider):
    """
    - For an azure model
    - if no base model set
    - don't enforce context window limits
    """
    try:
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": model,
                    "api_key": "my-fake-key",
                    "api_base": "my-fake-base",
                },
                "model_info": {"base_model": base_model, "id": "1"},
            }
        ]
        router = Router(
            model_list=model_list,
            set_verbose=True,
            enable_pre_call_checks=True,
            num_retries=0,
        )

        litellm.token_counter = MagicMock()

        def token_counter_side_effect(*args, **kwargs):
            # Process args and kwargs if needed
            return 1000000

        litellm.token_counter.side_effect = token_counter_side_effect
        try:
            updated_list = router._pre_call_checks(
                model="gpt-4",
                healthy_deployments=model_list,
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )
            if llm_provider == "azure" and base_model is None:
                assert len(updated_list) == 1
            else:
                pytest.fail("Expected to raise an error. Got={}".format(updated_list))
        except Exception as e:
            if (
                llm_provider == "azure" and base_model is not None
            ) or llm_provider == "openai":
                pass
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")


def test_router_cooldown_api_connection_error():
    try:
        _ = litellm.completion(
            model="vertex_ai/gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    except litellm.APIConnectionError as e:
        assert (
            Router()._is_cooldown_required(
                model_id="", exception_status=e.code, exception_str=str(e)
            )
            is False
        )

    router = Router(
        model_list=[
            {
                "model_name": "gemini-1.5-pro",
                "litellm_params": {"model": "vertex_ai/gemini-1.5-pro"},
            }
        ]
    )

    try:
        router.completion(
            model="gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    except litellm.APIConnectionError:
        pass

    try:
        router.completion(
            model="gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    except litellm.APIConnectionError:
        pass

    try:
        router.completion(
            model="gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    except litellm.APIConnectionError:
        pass


def test_router_correctly_reraise_error():
    """
    User feedback: There is a problem with my messages array, but the error exception thrown is a Rate Limit error.
    ```
    Rate Limit: Error code: 429 - {'error': {'message': 'No deployments available for selected model, Try again in 60 seconds. Passed model=gemini-1.5-flash..
    ```
    What they want? Propagation of the real error.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gemini-1.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-1.5-pro",
                    "mock_response": "litellm.RateLimitError",
                },
            }
        ]
    )

    try:
        router.completion(
            model="gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    except litellm.RateLimitError:
        pass


def test_router_dynamic_cooldown_correct_retry_after_time():
    """
    User feedback: litellm says "No deployments available for selected model, Try again in 60 seconds"
    but Azure says to retry in at most 9s

    ```
    {"message": "litellm.proxy.proxy_server.embeddings(): Exception occured - No deployments available for selected model, Try again in 60 seconds. Passed model=text-embedding-ada-002. pre-call-checks=False, allowed_model_region=n/a, cooldown_list=[('b49cbc9314273db7181fe69b1b19993f04efb88f2c1819947c538bac08097e4c', {'Exception Received': 'litellm.RateLimitError: AzureException RateLimitError - Requests to the Embeddings_Create Operation under Azure OpenAI API version 2023-09-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please retry after 9 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.', 'Status Code': '429'})]", "level": "ERROR", "timestamp": "2024-08-22T03:25:36.900476"}
    ```
    """
    router = Router(
        model_list=[
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "openai/text-embedding-ada-002",
                },
            }
        ]
    )

    openai_client = openai.OpenAI(api_key="")

    cooldown_time = 30.0

    def _return_exception(*args, **kwargs):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=429,
            detail="Rate Limited!",
            headers={"retry-after": cooldown_time},  # type: ignore
        )

    with patch.object(
        openai_client.embeddings.with_raw_response,
        "create",
        side_effect=_return_exception,
    ):
        new_retry_after_mock_client = MagicMock(return_value=-1)

        litellm.utils._get_retry_after_from_exception_header = (
            new_retry_after_mock_client
        )

        try:
            router.embedding(
                model="text-embedding-ada-002",
                input="Hello world!",
                client=openai_client,
            )
        except litellm.RateLimitError:
            pass

        new_retry_after_mock_client.assert_called()
        print(
            f"new_retry_after_mock_client.call_args.kwargs: {new_retry_after_mock_client.call_args.kwargs}"
        )

        response_headers: httpx.Headers = new_retry_after_mock_client.call_args.kwargs[
            "response_headers"
        ]
        assert "retry-after" in response_headers
        assert response_headers["retry-after"] == cooldown_time


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_aaarouter_dynamic_cooldown_message_retry_time(sync_mode):
    """
    User feedback: litellm says "No deployments available for selected model, Try again in 60 seconds"
    but Azure says to retry in at most 9s

    ```
    {"message": "litellm.proxy.proxy_server.embeddings(): Exception occured - No deployments available for selected model, Try again in 60 seconds. Passed model=text-embedding-ada-002. pre-call-checks=False, allowed_model_region=n/a, cooldown_list=[('b49cbc9314273db7181fe69b1b19993f04efb88f2c1819947c538bac08097e4c', {'Exception Received': 'litellm.RateLimitError: AzureException RateLimitError - Requests to the Embeddings_Create Operation under Azure OpenAI API version 2023-09-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please retry after 9 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.', 'Status Code': '429'})]", "level": "ERROR", "timestamp": "2024-08-22T03:25:36.900476"}
    ```
    """
    litellm.set_verbose = True
    router = Router(
        model_list=[
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "openai/text-embedding-ada-002",
                },
            },
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "openai/text-embedding-ada-002",
                },
            },
        ],
        set_verbose=True,
        debug_level="DEBUG",
    )

    openai_client = openai.OpenAI(api_key="")

    cooldown_time = 30.0

    def _return_exception(*args, **kwargs):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=429,
            detail="Rate Limited!",
            headers={"retry-after": cooldown_time},
        )

    with patch.object(
        openai_client.embeddings.with_raw_response,
        "create",
        side_effect=_return_exception,
    ):
        for _ in range(1):
            try:
                if sync_mode:
                    router.embedding(
                        model="text-embedding-ada-002",
                        input="Hello world!",
                        client=openai_client,
                    )
                else:
                    await router.aembedding(
                        model="text-embedding-ada-002",
                        input="Hello world!",
                        client=openai_client,
                    )
            except litellm.RateLimitError:
                pass

        if sync_mode:
            cooldown_deployments = _get_cooldown_deployments(
                litellm_router_instance=router
            )
        else:
            cooldown_deployments = await _async_get_cooldown_deployments(
                litellm_router_instance=router
            )
        print(
            "Cooldown deployments - {}\n{}".format(
                cooldown_deployments, len(cooldown_deployments)
            )
        )

        assert len(cooldown_deployments) > 0
        exception_raised = False
        try:
            if sync_mode:
                router.embedding(
                    model="text-embedding-ada-002",
                    input="Hello world!",
                    client=openai_client,
                )
            else:
                await router.aembedding(
                    model="text-embedding-ada-002",
                    input="Hello world!",
                    client=openai_client,
                )
        except litellm.types.router.RouterRateLimitError as e:
            print(e)
            exception_raised = True
            assert e.cooldown_time == cooldown_time

        assert exception_raised


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio()
async def test_router_weighted_pick(sync_mode):
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "weight": 2,
                    "mock_response": "Hello world 1!",
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "weight": 1,
                    "mock_response": "Hello world 2!",
                },
                "model_info": {"id": "2"},
            },
        ]
    )

    model_id_1_count = 0
    model_id_2_count = 0
    for _ in range(50):
        # make 50 calls. expect model id 1 to be picked more than model id 2
        if sync_mode:
            response = router.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello world!"}],
            )
        else:
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello world!"}],
            )

        model_id = int(response._hidden_params["model_id"])

        if model_id == 1:
            model_id_1_count += 1
        elif model_id == 2:
            model_id_2_count += 1
        else:
            raise Exception("invalid model id returned!")
    assert model_id_1_count > model_id_2_count


@pytest.mark.skip(reason="Hit azure batch quota limits")
@pytest.mark.parametrize("provider", ["azure"])
@pytest.mark.asyncio
async def test_router_batch_endpoints(provider):
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    print("Testing async create batch")

    router = Router(
        model_list=[
            {
                "model_name": "my-custom-name",
                "litellm_params": {
                    "model": "azure/gpt-4o-mini",
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_key": os.getenv("AZURE_API_KEY"),
                },
            },
        ]
    )

    file_name = "openai_batch_completions_router.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    file_obj = await router.acreate_file(
        model="my-custom-name",
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider=provider,
    )
    print("Response from creating file=", file_obj)

    await asyncio.sleep(10)
    batch_input_file_id = file_obj.id
    assert (
        batch_input_file_id is not None
    ), "Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    create_batch_response = await router.acreate_batch(
        model="my-custom-name",
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        custom_llm_provider=provider,
        metadata={"key1": "value1", "key2": "value2"},
    )

    print("response from router.create_batch=", create_batch_response)

    assert (
        create_batch_response.id is not None
    ), f"Failed to create batch, expected a non null batch_id but got {create_batch_response.id}"
    assert (
        create_batch_response.endpoint == "/v1/chat/completions"
        or create_batch_response.endpoint == "/chat/completions"
    ), f"Failed to create batch, expected endpoint to be /v1/chat/completions but got {create_batch_response.endpoint}"
    assert (
        create_batch_response.input_file_id == batch_input_file_id
    ), f"Failed to create batch, expected input_file_id to be {batch_input_file_id} but got {create_batch_response.input_file_id}"

    await asyncio.sleep(1)

    retrieved_batch = await router.aretrieve_batch(
        batch_id=create_batch_response.id,
        custom_llm_provider=provider,
    )
    print("retrieved batch=", retrieved_batch)
    # just assert that we retrieved a non None batch

    assert retrieved_batch.id == create_batch_response.id

    # list all batches
    list_batches = await router.alist_batches(
        model="my-custom-name", custom_llm_provider=provider, limit=2
    )
    print("list_batches=", list_batches)


@pytest.mark.parametrize("hidden", [True, False])
def test_model_group_alias(hidden):
    _model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
        {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
    ]
    router = Router(
        model_list=_model_list,
        model_group_alias={
            "gpt-4.5-turbo": {"model": "gpt-3.5-turbo", "hidden": hidden}
        },
    )

    models = router.get_model_list()

    model_names = router.get_model_names()

    if hidden:
        assert len(models) == len(_model_list)
        assert len(model_names) == len(_model_list)
    else:
        assert len(models) == len(_model_list) + 1
        assert len(model_names) == len(_model_list) + 1
