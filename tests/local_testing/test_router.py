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
from respx import MockRouter
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from litellm.types.router import ModelInfo
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
    litellm.set_verbose = True
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
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 1 = ", response1)

    response2 = await router.acompletion(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 2 = ", response2)

    response3 = await router.acompletion(
        model="groq/llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 3 = ", response3)

    response4 = await router.acompletion(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "hello"}],
    )


@pytest.mark.asyncio()
async def test_router_provider_wildcard_routing_regex():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/fo::*:static::*",
                "litellm_params": {
                    "model": "openai/fo::*:static::*",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
            },
            {
                "model_name": "openai/foo3::hello::*",
                "litellm_params": {
                    "model": "openai/foo3::hello::*",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
            },
        ]
    )

    print("router model list = ", router.get_model_list())

    response1 = await router.acompletion(
        model="openai/fo::anything-can-be-here::static::anything-can-be-here",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 1 = ", response1)

    response2 = await router.acompletion(
        model="openai/foo3::hello::static::anything-can-be-here",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 2 = ", response2)


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


@pytest.mark.skip(
    reason="Router no longer creates clients, this is delegated to the provider integration."
)
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


@pytest.mark.skip(
    reason="Router no longer creates clients, this is delegated to the provider integration."
)
def test_router_azure_ad_token_provider():
    _deployment = {
        "model_name": "gpt-4o_2024-05-13",
        "litellm_params": {
            "model": "azure/gpt-4o_2024-05-13",
            "api_base": "my-fake-route",
            "api_version": "2024-08-01-preview",
        },
        "model_info": {"id": "1234"},
    }
    for azure_cred in ["DefaultAzureCredential", "AzureCliCredential"]:
        os.environ["AZURE_CREDENTIAL"] = azure_cred
        litellm.enable_azure_ad_token_refresh = True
        router = Router(model_list=[_deployment])

        _client = router._get_client(
            deployment=_deployment,
            client_type="async",
            kwargs={"stream": False},
        )
        print(_client)
        import azure.identity as identity
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        assert isinstance(_client, AsyncOpenAI)
        assert isinstance(_client, AsyncAzureOpenAI)
        assert _client._azure_ad_token_provider is not None
        assert isinstance(_client._azure_ad_token_provider.__closure__, tuple)
        assert isinstance(
            _client._azure_ad_token_provider.__closure__[0].cell_contents._credential,
            getattr(identity, os.environ["AZURE_CREDENTIAL"]),
        )


def test_router_sensitive_keys():
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",  # openai model name
                    "litellm_params": {  # params for litellm completion/embedding call
                        "model": "azure/gpt-4.1-mini",
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
                "model": "gpt-4.1-nano",
                "api_key": os.getenv("OPENAI_API_KEY"),
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
@pytest.mark.skip(
    reason="Router no longer creates clients, this is delegated to the provider integration."
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

    # model_client = router._get_client(
    #     deployment={"model_info": {"id": 1234}}, client_type="sync_client", kwargs={}
    # )
    # url = getattr(model_client, "_base_url")
    # uri_reference = str(getattr(url, "_uri_reference"))

    # print(f"uri_reference: {uri_reference}")

    # assert "/v1/" in uri_reference
    # assert uri_reference.count("v1") == 1
    response = router.completion(
        model="azure/mistral-large-latest",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )
    assert response is not None


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
                    "model": "azure/gpt-4.1-mini",
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
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
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
        router.reset()
    except Exception as e:
        print(f"FAILED TEST")
        pytest.fail(f"Got unexpected exception on router! - {e}")


# test_reading_key_from_model_list()


def test_call_one_endpoint():
    # [PROD TEST CASE]
    # user passes one deployment they want to call on the router, we call the specified one
    # this test makes a completion calls azure/gpt-4.1-mini, it should work
    try:
        print("Testing calling a specific deployment")
        old_api_key = os.environ["AZURE_API_KEY"]

        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
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
                    "model": "azure/text-embedding-ada-002",
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
                model="azure/gpt-4.1-mini",
                messages=[{"role": "user", "content": "hello this request will pass"}],
                specific_deployment=True,
            )
            print("\n response", response)

        async def call_azure_embedding():
            response = await router.aembedding(
                model="azure/text-embedding-ada-002",
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
                    "model": "azure/gpt-4.1-mini",
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


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_async_router_context_window_fallback(sync_mode):
    """
    - Give a gpt-4 model group with different context windows (8192k vs. 128k)
    - Send a 10k prompt
    - Assert it works
    """
    import os

    from large_text import text

    litellm.set_verbose = False
    litellm._turn_on_debug()

    print(f"len(text): {len(text)}")
    try:
        model_list = [
            {
                "model_name": "gpt-4",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "api_base": os.getenv("OPENAI_API_BASE"),
                },
            },
            {
                "model_name": "gpt-4-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-4-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, context_window_fallbacks=[{"gpt-4": ["gpt-4-turbo"]}], num_retries=0)  # type: ignore
        if sync_mode is False:
            response = await router.acompletion(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": text * 2},
                    {"role": "user", "content": "Who was Alexander?"},
                ],
            )

            print(f"response: {response}")
            assert "gpt-4-turbo" in response.model
        else:
            response = router.completion(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": text * 2},
                    {"role": "user", "content": "Who was Alexander?"},
                ],
            )
            assert "gpt-4-turbo" in response.model
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
        except Exception:
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
                    "model": "azure/gpt-4.1-mini",
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
                    "model": "azure/gpt-4.1-mini",
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
                    "model": "azure/gpt-4.1-mini",
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


@pytest.mark.parametrize("allowed_model_region", ["eu", None, "us"])
def test_router_region_pre_call_check(allowed_model_region):
    """
    If region based routing set
    - check if only model in allowed region is allowed by '_pre_call_checks'
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",  # openai model name
            "litellm_params": {  # params for litellm completion/embedding call
                "model": "azure/gpt-4.1-mini",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "base_model": "azure/gpt-35-turbo",
                "region_name": allowed_model_region,
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
        assert len(_healthy_deployments) == 1, "{} models selected as healthy".format(
            len(_healthy_deployments)
        )
        assert (
            _healthy_deployments[0]["model_info"]["id"] == "1"
        ), "Incorrect model id picked. Got id={}, expected id=1".format(
            _healthy_deployments[0]["model_info"]["id"]
        )


### FUNCTION CALLING


def test_function_calling():
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
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
        model="gpt-3.5-turbo", messages=messages, functions=functions
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
                    "model": "gpt-3.5-turbo",
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
            }
        ]
        router = Router(model_list=model_list, num_retries=3)
        response = await router.aimage_generation(
            model="dall-e-3", prompt="A cute baby sea otter"
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
            }
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
            ## Test 1: user facing function
            response = await router.aembedding(
                model="text-embedding-ada-002",
                input=["good morning from litellm", "this is another item"],
            )
            print(response)

            ## Test 2: underlying function
            response = await router._aembedding(
                model="text-embedding-ada-002",
                input=["good morning from litellm 2"],
            )
            print(response)
            router.reset()

        asyncio.run(embedding_call())

        print("\n Making sync Embedding call\n")
        ## Test 1: user facing function
        response = router.embedding(
            model="text-embedding-ada-002",
            input=["good morning from litellm 2"],
        )
        print(response)
        router.reset()

        ## Test 2: underlying function
        response = router._embedding(
            model="text-embedding-ada-002",
            input=["good morning from litellm 2"],
        )
        print(response)
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
                    "model": "azure/text-embedding-ada-002",
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


# test_bedrock_on_router()


# test openai-compatible endpoint
@pytest.mark.asyncio
async def test_mistral_on_router():
    litellm._turn_on_debug()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "mistral/mistral-small-latest",
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
                "model": "claude-3-5-haiku-20241022",
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
                "model": "omni-moderation-latest",
                "api_key": os.getenv("OPENAI_API_KEY", None),
            },
        }
    ]

    router = Router(model_list=model_list)
    ## Test 1: user facing function
    result = await router.amoderation(
        model="omni-moderation-latest", input="this is valid good text"
    )


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
        router.get_router_model_info(
            deployment=deployment.to_json(), received_model_name=model
        )
    else:
        # Azure models without base_model now fallback to using the original model name
        # instead of raising an exception. This should succeed but log a warning.
        model_info = router.get_router_model_info(
            deployment=deployment.to_json(), received_model_name=model
        )
        # Verify that model_info is returned (even if it may have default values)
        assert model_info is not None


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
    from litellm.router_utils.cooldown_handlers import _is_cooldown_required

    try:
        _ = litellm.completion(
            model="vertex_ai/gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    except litellm.APIConnectionError as e:
        assert (
            _is_cooldown_required(
                litellm_router_instance=Router(),
                model_id="",
                exception_status=e.code,
                exception_str=str(e),
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


def test_router_correctly_reraise_error():
    """
    User feedback: There is a problem with my messages array, but the error exception thrown is a Rate Limit error.
    ```
    Rate Limit: Error code: 429 - {'error': {'message': 'No deployments available for selected model, Try again in 60 seconds. Passed model=gemini-2.5-flash-lite..
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

    cooldown_time = 30

    def _return_exception(*args, **kwargs):
        from httpx import Headers, Request, Response

        kwargs = {
            "request": Request("POST", "https://www.google.com"),
            "message": "Error code: 429 - Rate Limit Error!",
            "body": {"detail": "Rate Limit Error!"},
            "code": None,
            "param": None,
            "type": None,
            "response": Response(
                status_code=429,
                headers=Headers(
                    {
                        "date": "Sat, 21 Sep 2024 22:56:53 GMT",
                        "server": "uvicorn",
                        "retry-after": f"{cooldown_time}",
                        "content-length": "30",
                        "content-type": "application/json",
                    }
                ),
                request=Request("POST", "http://0.0.0.0:9000/chat/completions"),
            ),
            "status_code": 429,
            "request_id": None,
        }

        exception = Exception()
        for k, v in kwargs.items():
            setattr(exception, k, v)
        raise exception

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

        response_headers: httpx.Headers = new_retry_after_mock_client.call_args[0][0]
        assert int(response_headers["retry-after"]) == cooldown_time


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
    cooldown_time = 30.0
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
        cooldown_time=cooldown_time,
    )

    openai_client = openai.OpenAI(api_key="")

    def _return_exception(*args, **kwargs):
        from httpx import Headers, Request, Response

        kwargs = {
            "request": Request("POST", "https://www.google.com"),
            "message": "Error code: 429 - Rate Limit Error!",
            "body": {"detail": "Rate Limit Error!"},
            "code": None,
            "param": None,
            "type": None,
            "response": Response(
                status_code=429,
                headers=Headers(
                    {
                        "date": "Sat, 21 Sep 2024 22:56:53 GMT",
                        "server": "uvicorn",
                        "retry-after": f"{cooldown_time}",
                        "content-length": "30",
                        "content-type": "application/json",
                    }
                ),
                request=Request("POST", "http://0.0.0.0:9000/chat/completions"),
            ),
            "status_code": 429,
            "request_id": None,
        }

        exception = Exception()
        for k, v in kwargs.items():
            setattr(exception, k, v)
        raise exception

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

        await asyncio.sleep(5)

        if sync_mode:
            cooldown_deployments = _get_cooldown_deployments(
                litellm_router_instance=router, parent_otel_span=None
            )
        else:
            cooldown_deployments = await _async_get_cooldown_deployments(
                litellm_router_instance=router, parent_otel_span=None
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
@pytest.mark.flaky(retries=6, delay=1)
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

    ## TEST 2 - test underlying create_file function
    file_obj = await router._acreate_file(
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
    ## TEST 2 - test underlying create_batch function
    create_batch_response = await router._acreate_batch(
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


def test_get_team_specific_model():
    """
    Test that _get_team_specific_model returns:
    - team_public_model_name when team_id matches
    - None when team_id doesn't match
    - None when no team_id in model_info
    """
    router = Router(model_list=[])

    # Test 1: Matching team_id
    deployment = DeploymentTypedDict(
        model_name="model-x",
        litellm_params={},
        model_info=ModelInfo(team_id="team1", team_public_model_name="public-model-x"),
    )
    assert router._get_team_specific_model(deployment, "team1") == "public-model-x"

    # Test 2: Non-matching team_id
    assert router._get_team_specific_model(deployment, "team2") is None

    # Test 3: No team_id in model_info
    deployment = DeploymentTypedDict(
        model_name="model-y",
        litellm_params={},
        model_info=ModelInfo(team_public_model_name="public-model-y"),
    )
    assert router._get_team_specific_model(deployment, "team1") is None

    # Test 4: No model_info
    deployment = DeploymentTypedDict(
        model_name="model-z", litellm_params={}, model_info=ModelInfo()
    )
    assert router._get_team_specific_model(deployment, "team1") is None


def test_is_team_specific_model():
    """
    Test that _is_team_specific_model returns:
    - True when model_info contains team_id
    - False when model_info doesn't contain team_id
    - False when model_info is None
    """
    router = Router(model_list=[])

    # Test 1: With team_id
    model_info = ModelInfo(team_id="team1", team_public_model_name="public-model-x")
    assert router._is_team_specific_model(model_info) is True

    # Test 2: Without team_id
    model_info = ModelInfo(team_public_model_name="public-model-y")
    assert router._is_team_specific_model(model_info) is False

    # Test 3: Empty model_info
    model_info = ModelInfo()
    assert router._is_team_specific_model(model_info) is False

    # Test 4: None model_info
    assert router._is_team_specific_model(None) is False


# @pytest.mark.parametrize("on_error", [True, False])
# @pytest.mark.asyncio
# async def test_router_response_headers(on_error):
#     router = Router(
#         model_list=[
#             {
#                 "model_name": "gpt-3.5-turbo",
#                 "litellm_params": {
#                     "model": "azure/gpt-4.1-mini",
#                     "api_key": os.getenv("AZURE_API_KEY"),
#                     "api_base": os.getenv("AZURE_API_BASE"),
#                     "tpm": 100000,
#                     "rpm": 100000,
#                 },
#             },
#             {
#                 "model_name": "gpt-3.5-turbo",
#                 "litellm_params": {
#                     "model": "azure/gpt-4.1-mini",
#                     "api_key": os.getenv("AZURE_API_KEY"),
#                     "api_base": os.getenv("AZURE_API_BASE"),
#                     "tpm": 500,
#                     "rpm": 500,
#                 },
#             },
#         ]
#     )

#     response = await router.acompletion(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": "Hello world!"}],
#         mock_testing_rate_limit_error=on_error,
#     )

#     response_headers = response._hidden_params["additional_headers"]

#     print(response_headers)

#     assert response_headers["x-ratelimit-limit-requests"] == 100500
#     assert int(response_headers["x-ratelimit-remaining-requests"]) > 0
#     assert response_headers["x-ratelimit-limit-tokens"] == 100500
#     assert int(response_headers["x-ratelimit-remaining-tokens"]) > 0


def test_router_completion_with_model_id():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "123"},
            }
        ]
    )

    with patch.object(
        router, "routing_strategy_pre_call_checks"
    ) as mock_pre_call_checks:
        router.completion(model="123", messages=[{"role": "user", "content": "hi"}])
        mock_pre_call_checks.assert_not_called()


def test_router_prompt_management_factory():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
            {
                "model_name": "chatbot_actions",
                "litellm_params": {
                    "model": "langfuse/openai-gpt-3.5-turbo",
                    "tpm": 1000000,
                    "prompt_id": "jokes",
                },
            },
            {
                "model_name": "openai-gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]
    )

    assert router._is_prompt_management_model("chatbot_actions") is True
    assert router._is_prompt_management_model("openai-gpt-3.5-turbo") is False

    response = router._prompt_management_factory(
        model="chatbot_actions",
        messages=[{"role": "user", "content": "Hello world!"}],
        kwargs={},
    )

    print(response)


def test_router_get_model_list_from_model_alias():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        model_group_alias={
            "my-special-fake-model-alias-name": "fake-openai-endpoint-3"
        },
    )

    model_alias_list = router.get_model_list_from_model_alias(
        model_name="gpt-3.5-turbo"
    )
    assert len(model_alias_list) == 0


def test_router_dynamic_credentials():
    """
    Assert model id for dynamic api key 1 != model id for dynamic api key 2
    """
    original_model_id = "123"
    original_api_key = "my-bad-key"
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": original_api_key,
                    "mock_response": "fake_response",
                },
                "model_info": {"id": original_model_id},
            }
        ]
    )

    deployment = router.get_deployment(model_id=original_model_id)
    assert deployment is not None
    assert deployment.litellm_params.api_key == original_api_key

    response = router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        api_key="my-bad-key-2",
    )

    response_2 = router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        api_key="my-bad-key-3",
    )

    assert response_2._hidden_params["model_id"] != response._hidden_params["model_id"]

    deployment = router.get_deployment(model_id=original_model_id)
    assert deployment is not None
    assert deployment.litellm_params.api_key == original_api_key


def test_router_get_model_group_info():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
            },
        ],
    )

    model_group_info = router.get_model_group_info(model_group="gpt-4")
    assert model_group_info is not None
    assert model_group_info.model_group == "gpt-4"
    assert model_group_info.input_cost_per_token > 0
    assert model_group_info.output_cost_per_token > 0
