# this tests if the router is initialized correctly
import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

import litellm
from litellm import Router
from litellm.router_utils.client_initalization_utils import (
    InitalizeOpenAISDKClient,
    OpenAISDKClientInitializationParams,
)

load_dotenv()

# every time we load the router we should have 4 clients:
# Async
# Sync
# Async + Stream
# Sync + Stream


def test_init_clients():
    litellm.set_verbose = True
    import logging

    from litellm._logging import verbose_router_logger

    verbose_router_logger.setLevel(logging.DEBUG)
    try:
        print("testing init 4 clients with diff timeouts")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "timeout": 0.01,
                    "stream_timeout": 0.000_001,
                    "max_retries": 7,
                },
            },
        ]
        router = Router(model_list=model_list, set_verbose=True)
        for elem in router.model_list:
            model_id = elem["model_info"]["id"]
            assert router.cache.get_cache(f"{model_id}_client") is not None
            assert router.cache.get_cache(f"{model_id}_async_client") is not None
            assert router.cache.get_cache(f"{model_id}_stream_client") is not None
            assert router.cache.get_cache(f"{model_id}_stream_async_client") is not None

            # check if timeout for stream/non stream clients is set correctly
            async_client = router.cache.get_cache(f"{model_id}_async_client")
            stream_async_client = router.cache.get_cache(
                f"{model_id}_stream_async_client"
            )

            assert async_client.timeout == 0.01
            assert stream_async_client.timeout == 0.000_001
            print(vars(async_client))
            print()
            print(async_client._base_url)
            assert (
                async_client._base_url
                == "https://openai-gpt-4-test-v-1.openai.azure.com//openai/"
            )  # openai python adds the extra /
            assert (
                stream_async_client._base_url
                == "https://openai-gpt-4-test-v-1.openai.azure.com//openai/"
            )

        print("PASSED !")

    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_init_clients()


def test_init_clients_basic():
    litellm.set_verbose = True
    try:
        print("Test basic client init")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            },
        ]
        router = Router(model_list=model_list)
        for elem in router.model_list:
            model_id = elem["model_info"]["id"]
            assert router.cache.get_cache(f"{model_id}_client") is not None
            assert router.cache.get_cache(f"{model_id}_async_client") is not None
            assert router.cache.get_cache(f"{model_id}_stream_client") is not None
            assert router.cache.get_cache(f"{model_id}_stream_async_client") is not None
        print("PASSED !")

        # see if we can init clients without timeout or max retries set
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_init_clients_basic()


def test_init_clients_basic_azure_cloudflare():
    # init azure + cloudflare
    # init OpenAI gpt-3.5
    # init OpenAI text-embedding
    # init OpenAI comptaible - Mistral/mistral-medium
    # init OpenAI compatible - xinference/bge
    litellm.set_verbose = True
    try:
        print("Test basic client init")
        model_list = [
            {
                "model_name": "azure-cloudflare",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": "https://gateway.ai.cloudflare.com/v1/0399b10e77ac6668c80404a5ff49eb37/litellm-test/azure-openai/openai-gpt-4-test-v-1",
                },
            },
            {
                "model_name": "gpt-openai",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "mistral",
                "litellm_params": {
                    "model": "mistral/mistral-tiny",
                    "api_key": os.getenv("MISTRAL_API_KEY"),
                },
            },
            {
                "model_name": "bge-base-en",
                "litellm_params": {
                    "model": "xinference/bge-base-en",
                    "api_base": "http://127.0.0.1:9997/v1",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]
        router = Router(model_list=model_list)
        for elem in router.model_list:
            model_id = elem["model_info"]["id"]
            assert router.cache.get_cache(f"{model_id}_client") is not None
            assert router.cache.get_cache(f"{model_id}_async_client") is not None
            assert router.cache.get_cache(f"{model_id}_stream_client") is not None
            assert router.cache.get_cache(f"{model_id}_stream_async_client") is not None
        print("PASSED !")

        # see if we can init clients without timeout or max retries set
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_init_clients_basic_azure_cloudflare()


def test_timeouts_router():
    """
    Test the timeouts of the router with multiple clients. This HASas to raise a timeout error
    """
    import openai

    litellm.set_verbose = True
    try:
        print("testing init 4 clients with diff timeouts")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "timeout": 0.000001,
                    "stream_timeout": 0.000_001,
                },
            },
        ]
        router = Router(model_list=model_list, num_retries=0)

        print("PASSED !")

        async def test():
            try:
                await router.acompletion(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "hello, write a 20 pg essay"}
                    ],
                )
            except Exception as e:
                raise e

        asyncio.run(test())
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


# test_timeouts_router()


def test_stream_timeouts_router():
    """
    Test the stream timeouts router. See if it selected the correct client with stream timeout
    """
    import openai

    litellm.set_verbose = True
    try:
        print("testing init 4 clients with diff timeouts")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "timeout": 200,  # regular calls will not timeout, stream calls will
                    "stream_timeout": 10,
                },
            },
        ]
        router = Router(model_list=model_list)

        print("PASSED !")
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello, write a 20 pg essay"}],
            "stream": True,
        }
        selected_client = router._get_client(
            deployment=router.model_list[0],
            kwargs=data,
            client_type=None,
        )
        print("Select client timeout", selected_client.timeout)
        assert selected_client.timeout == 10

        # make actual call
        response = router.completion(**data)

        for chunk in response:
            print(f"chunk: {chunk}")
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


# test_stream_timeouts_router()


def test_xinference_embedding():
    # [Test Init Xinference] this tests if we init xinference on the router correctly
    # [Test Exception Mapping] tests that xinference is an openai comptiable provider
    print("Testing init xinference")
    print(
        "this tests if we create an OpenAI client for Xinference, with the correct API BASE"
    )

    model_list = [
        {
            "model_name": "xinference",
            "litellm_params": {
                "model": "xinference/bge-base-en",
                "api_base": "os.environ/XINFERENCE_API_BASE",
            },
        }
    ]

    router = Router(model_list=model_list)

    print(router.model_list)
    print(router.model_list[0])

    assert (
        router.model_list[0]["litellm_params"]["api_base"] == "http://0.0.0.0:9997"
    )  # set in env

    openai_client = router._get_client(
        deployment=router.model_list[0],
        kwargs={"input": ["hello"], "model": "xinference"},
    )

    assert openai_client._base_url == "http://0.0.0.0:9997"
    assert "xinference" in litellm.openai_compatible_providers
    print("passed")


# test_xinference_embedding()


def test_router_init_gpt_4_vision_enhancements():
    try:
        # tests base_url set when any base_url with /openai/deployments passed to router
        print("Testing Azure GPT_Vision enhancements")

        model_list = [
            {
                "model_name": "gpt-4-vision-enhancements",
                "litellm_params": {
                    "model": "azure/gpt-4-vision",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "base_url": "https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions/",
                    "dataSources": [
                        {
                            "type": "AzureComputerVision",
                            "parameters": {
                                "endpoint": "os.environ/AZURE_VISION_ENHANCE_ENDPOINT",
                                "key": "os.environ/AZURE_VISION_ENHANCE_KEY",
                            },
                        }
                    ],
                },
            }
        ]

        router = Router(model_list=model_list)

        print(router.model_list)
        print(router.model_list[0])

        assert (
            router.model_list[0]["litellm_params"]["base_url"]
            == "https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions/"
        )  # set in env

        assert (
            router.model_list[0]["litellm_params"]["dataSources"][0]["parameters"][
                "endpoint"
            ]
            == os.environ["AZURE_VISION_ENHANCE_ENDPOINT"]
        )

        assert (
            router.model_list[0]["litellm_params"]["dataSources"][0]["parameters"][
                "key"
            ]
            == os.environ["AZURE_VISION_ENHANCE_KEY"]
        )

        azure_client = router._get_client(
            deployment=router.model_list[0],
            kwargs={"stream": True, "model": "gpt-4-vision-enhancements"},
            client_type="async",
        )

        assert (
            azure_client._base_url
            == "https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions/"
        )
        print("passed")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_with_organization(sync_mode):
    try:
        print("Testing OpenAI with organization")
        model_list = [
            {
                "model_name": "openai-bad-org",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "organization": "org-ikDc4ex8NB",
                },
            },
            {
                "model_name": "openai-good-org",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
        ]

        router = Router(model_list=model_list)

        print(router.model_list)
        print(router.model_list[0])

        if sync_mode:
            openai_client = router._get_client(
                deployment=router.model_list[0],
                kwargs={"input": ["hello"], "model": "openai-bad-org"},
            )
            print(vars(openai_client))

            assert openai_client.organization == "org-ikDc4ex8NB"

            # bad org raises error

            try:
                response = router.completion(
                    model="openai-bad-org",
                    messages=[{"role": "user", "content": "this is a test"}],
                )
                pytest.fail(
                    "Request should have failed - This organization does not exist"
                )
            except Exception as e:
                print("Got exception: " + str(e))
                assert "No such organization: org-ikDc4ex8NB" in str(e)

            # good org works
            response = router.completion(
                model="openai-good-org",
                messages=[{"role": "user", "content": "this is a test"}],
                max_tokens=5,
            )
        else:
            openai_client = router._get_client(
                deployment=router.model_list[0],
                kwargs={"input": ["hello"], "model": "openai-bad-org"},
                client_type="async",
            )
            print(vars(openai_client))

            assert openai_client.organization == "org-ikDc4ex8NB"

            # bad org raises error

            try:
                response = await router.acompletion(
                    model="openai-bad-org",
                    messages=[{"role": "user", "content": "this is a test"}],
                )
                pytest.fail(
                    "Request should have failed - This organization does not exist"
                )
            except Exception as e:
                print("Got exception: " + str(e))
                assert "No such organization: org-ikDc4ex8NB" in str(e)

            # good org works
            response = await router.acompletion(
                model="openai-good-org",
                messages=[{"role": "user", "content": "this is a test"}],
                max_tokens=5,
            )

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_init_clients_azure_command_r_plus():
    # This tests that the router uses the OpenAI client for Azure/Command-R+
    # For azure/command-r-plus we need to use openai.OpenAI because of how the Azure provider requires requests being sent
    litellm.set_verbose = True
    import logging

    from litellm._logging import verbose_router_logger

    verbose_router_logger.setLevel(logging.DEBUG)
    try:
        print("testing init 4 clients with diff timeouts")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/command-r-plus",
                    "api_key": os.getenv("AZURE_COHERE_API_KEY"),
                    "api_base": os.getenv("AZURE_COHERE_API_BASE"),
                    "timeout": 0.01,
                    "stream_timeout": 0.000_001,
                    "max_retries": 7,
                },
            },
        ]
        router = Router(model_list=model_list, set_verbose=True)
        for elem in router.model_list:
            model_id = elem["model_info"]["id"]
            async_client = router.cache.get_cache(f"{model_id}_async_client")
            stream_async_client = router.cache.get_cache(
                f"{model_id}_stream_async_client"
            )
            # Assert the Async Clients used are OpenAI clients and not Azure
            # For using Azure/Command-R-Plus and Azure/Mistral the clients NEED to be OpenAI clients used
            # this is weirdness introduced on Azure's side

            assert "openai.AsyncOpenAI" in str(async_client)
            assert "openai.AsyncOpenAI" in str(stream_async_client)
        print("PASSED !")

    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_text_completion_with_organization():
    try:
        print("Testing Text OpenAI with organization")
        model_list = [
            {
                "model_name": "openai-bad-org",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": os.getenv("OPENAI_API_KEY", None),
                    "organization": "org-ikDc4ex8NB",
                },
            },
            {
                "model_name": "openai-good-org",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": os.getenv("OPENAI_API_KEY", None),
                    "organization": os.getenv("OPENAI_ORGANIZATION", None),
                },
            },
        ]

        router = Router(model_list=model_list)

        print(router.model_list)
        print(router.model_list[0])

        openai_client = router._get_client(
            deployment=router.model_list[0],
            kwargs={"input": ["hello"], "model": "openai-bad-org"},
        )
        print(vars(openai_client))

        assert openai_client.organization == "org-ikDc4ex8NB"

        # bad org raises error

        try:
            response = await router.atext_completion(
                model="openai-bad-org",
                prompt="this is a test",
            )
            pytest.fail("Request should have failed - This organization does not exist")
        except Exception as e:
            print("Got exception: " + str(e))
            assert "No such organization: org-ikDc4ex8NB" in str(e)

        # good org works
        response = await router.atext_completion(
            model="openai-good-org",
            prompt="this is a test",
            max_tokens=5,
        )
        print("working response: ", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_init_clients_async_mode():
    litellm.set_verbose = True
    import logging

    from litellm._logging import verbose_router_logger
    from litellm.types.router import RouterGeneralSettings

    verbose_router_logger.setLevel(logging.DEBUG)
    try:
        print("testing init 4 clients with diff timeouts")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "timeout": 0.01,
                    "stream_timeout": 0.000_001,
                    "max_retries": 7,
                },
            },
        ]
        router = Router(
            model_list=model_list,
            set_verbose=True,
            router_general_settings=RouterGeneralSettings(async_only_mode=True),
        )
        for elem in router.model_list:
            model_id = elem["model_info"]["id"]

            # sync clients not initialized in async_only_mode=True
            assert router.cache.get_cache(f"{model_id}_client") is None
            assert router.cache.get_cache(f"{model_id}_stream_client") is None

            # only async clients initialized in async_only_mode=True
            assert router.cache.get_cache(f"{model_id}_async_client") is not None
            assert router.cache.get_cache(f"{model_id}_stream_async_client") is not None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "environment,expected_models",
    [
        ("development", ["gpt-3.5-turbo"]),
        ("production", ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]),
    ],
)
def test_init_router_with_supported_environments(environment, expected_models):
    """
    Tests that the correct models are setup on router when LITELLM_ENVIRONMENT is set
    """
    os.environ["LITELLM_ENVIRONMENT"] = environment
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-2",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "timeout": 0.01,
                "stream_timeout": 0.000_001,
                "max_retries": 7,
            },
            "model_info": {"supported_environments": ["development", "production"]},
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "timeout": 0.01,
                "stream_timeout": 0.000_001,
                "max_retries": 7,
            },
            "model_info": {"supported_environments": ["production"]},
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "timeout": 0.01,
                "stream_timeout": 0.000_001,
                "max_retries": 7,
            },
            "model_info": {"supported_environments": ["production"]},
        },
    ]
    router = Router(model_list=model_list, set_verbose=True)
    _model_list = router.get_model_names()

    print("model_list: ", _model_list)
    print("expected_models: ", expected_models)

    assert set(_model_list) == set(expected_models)

    os.environ.pop("LITELLM_ENVIRONMENT")


def test_should_initialize_sync_client():
    from litellm.types.router import RouterGeneralSettings

    # Test case 1: Router instance is None
    assert InitalizeOpenAISDKClient.should_initialize_sync_client(None) is False

    # Test case 2: Router instance without router_general_settings
    router = Router(model_list=[])
    assert InitalizeOpenAISDKClient.should_initialize_sync_client(router) is True

    # Test case 3: Router instance with async_only_mode = False
    router = Router(
        model_list=[],
        router_general_settings=RouterGeneralSettings(async_only_mode=False),
    )
    assert InitalizeOpenAISDKClient.should_initialize_sync_client(router) is True

    # Test case 4: Router instance with async_only_mode = True
    router = Router(
        model_list=[],
        router_general_settings=RouterGeneralSettings(async_only_mode=True),
    )
    assert InitalizeOpenAISDKClient.should_initialize_sync_client(router) is False

    # Test case 5: Router instance with router_general_settings but without async_only_mode
    router = Router(model_list=[], router_general_settings=RouterGeneralSettings())
    assert InitalizeOpenAISDKClient.should_initialize_sync_client(router) is True

    print("All test cases passed!")


@pytest.mark.parametrize(
    "model_name, custom_llm_provider, expected_result",
    [
        ("gpt-3.5-turbo", None, True),  # OpenAI chat completion model
        ("text-embedding-ada-002", None, True),  # OpenAI embedding model
        ("claude-2", None, False),  # Non-OpenAI model
        ("gpt-3.5-turbo", "azure", True),  # Azure OpenAI
        ("text-davinci-003", "azure_text", True),  # Azure OpenAI
        ("gpt-3.5-turbo", "openai", True),  # OpenAI
        ("custom-model", "custom_openai", True),  # Custom OpenAI compatible
        ("text-davinci-003", "text-completion-openai", True),  # OpenAI text completion
        (
            "ft:gpt-3.5-turbo-0613:my-org:custom-model:7p4lURel",
            None,
            True,
        ),  # Fine-tuned GPT model
        ("mistral-7b", "huggingface", False),  # Non-OpenAI provider
        ("custom-model", "anthropic", False),  # Non-OpenAI compatible provider
    ],
)
def test_should_create_openai_sdk_client_for_model(
    model_name, custom_llm_provider, expected_result
):
    result = InitalizeOpenAISDKClient._should_create_openai_sdk_client_for_model(
        model_name, custom_llm_provider
    )
    assert (
        result == expected_result
    ), f"Failed for model: {model_name}, provider: {custom_llm_provider}"


def test_should_create_openai_sdk_client_for_model_openai_compatible_providers():
    # Test with a known OpenAI compatible provider
    assert InitalizeOpenAISDKClient._should_create_openai_sdk_client_for_model(
        "custom-model", "groq"
    ), "Should return True for OpenAI compatible provider"

    # Add a new compatible provider and test
    litellm.openai_compatible_providers.append("new_provider")
    assert InitalizeOpenAISDKClient._should_create_openai_sdk_client_for_model(
        "custom-model", "new_provider"
    ), "Should return True for newly added OpenAI compatible provider"

    # Clean up
    litellm.openai_compatible_providers.remove("new_provider")


def test_get_client_initialization_params_openai():
    """Test basic OpenAI configuration with direct parameter passing."""
    model = {}
    model_name = "gpt-3.5-turbo"
    custom_llm_provider = None
    litellm_params = {"api_key": "sk-openai-key", "timeout": 30, "max_retries": 3}
    default_api_key = None
    default_api_base = None

    result = InitalizeOpenAISDKClient._get_client_initialization_params(
        model=model,
        model_name=model_name,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        default_api_key=default_api_key,
        default_api_base=default_api_base,
    )

    assert isinstance(result, OpenAISDKClientInitializationParams)
    assert result.api_key == "sk-openai-key"
    assert result.timeout == 30
    assert result.max_retries == 3
    assert result.model_name == "gpt-3.5-turbo"


def test_get_client_initialization_params_azure():
    """Test Azure OpenAI configuration with specific Azure parameters."""
    model = {}
    model_name = "azure/gpt-4"
    custom_llm_provider = "azure"
    litellm_params = {
        "api_key": "azure-key",
        "api_base": "https://example.azure.openai.com",
        "api_version": "2023-05-15",
    }
    default_api_key = None
    default_api_base = None

    result = InitalizeOpenAISDKClient._get_client_initialization_params(
        model=model,
        model_name=model_name,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        default_api_key=default_api_key,
        default_api_base=default_api_base,
    )

    assert result.api_key == "azure-key"
    assert result.api_base == "https://example.azure.openai.com"
    assert result.api_version == "2023-05-15"
    assert result.custom_llm_provider == "azure"


def test_get_client_initialization_params_environment_variable_parsing():
    """Test parsing of environment variables for configuration."""
    os.environ["UNIQUE_OPENAI_API_KEY"] = "env-openai-key"
    os.environ["UNIQUE_TIMEOUT"] = "45"

    model = {}
    model_name = "gpt-4"
    custom_llm_provider = None
    litellm_params = {
        "api_key": "os.environ/UNIQUE_OPENAI_API_KEY",
        "timeout": "os.environ/UNIQUE_TIMEOUT",
        "organization": "os.environ/UNIQUE_ORG_ID",
    }
    default_api_key = None
    default_api_base = None

    result = InitalizeOpenAISDKClient._get_client_initialization_params(
        model=model,
        model_name=model_name,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        default_api_key=default_api_key,
        default_api_base=default_api_base,
    )

    assert result.api_key == "env-openai-key"
    assert result.timeout == 45.0
    assert result.organization is None  # Since ORG_ID is not set in the environment


def test_get_client_initialization_params_azure_ai_studio_mistral():
    """
    Test configuration for Azure AI Studio Mistral model.

    - /v1/ is added to the api_base if it is not present
    - custom_llm_provider is set to openai (Azure AI Studio Mistral models need to use OpenAI route)
    """

    model = {}
    model_name = "azure/mistral-large-latest"
    custom_llm_provider = "azure"
    litellm_params = {
        "api_key": "azure-key",
        "api_base": "https://example.azure.openai.com",
    }
    default_api_key = None
    default_api_base = None

    result = InitalizeOpenAISDKClient._get_client_initialization_params(
        model,
        model_name,
        custom_llm_provider,
        litellm_params,
        default_api_key,
        default_api_base,
    )

    assert result.custom_llm_provider == "openai"
    assert result.model_name == "mistral-large-latest"
    assert result.api_base == "https://example.azure.openai.com/v1/"


def test_get_client_initialization_params_default_values():
    """
    Test use of default values when specific parameters are not provided.

    This is used typically for OpenAI compatible providers - example Together AI

    """
    model = {}
    model_name = "together/meta-llama-3.1-8b-instruct"
    custom_llm_provider = None
    litellm_params = {}
    default_api_key = "together-api-key"
    default_api_base = "https://together.xyz/api.openai.com"

    result = InitalizeOpenAISDKClient._get_client_initialization_params(
        model=model,
        model_name=model_name,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        default_api_key=default_api_key,
        default_api_base=default_api_base,
    )

    assert result.api_key == "together-api-key"
    assert result.api_base == "https://together.xyz/api.openai.com"
    assert result.timeout == litellm.request_timeout
    assert result.max_retries == 0


def test_get_client_initialization_params_all_env_vars():
    # Set up environment variables
    os.environ["TEST_API_KEY"] = "test-api-key"
    os.environ["TEST_API_BASE"] = "https://test.openai.com"
    os.environ["TEST_API_VERSION"] = "2023-05-15"
    os.environ["TEST_TIMEOUT"] = "30"
    os.environ["TEST_STREAM_TIMEOUT"] = "60"
    os.environ["TEST_MAX_RETRIES"] = "3"
    os.environ["TEST_ORGANIZATION"] = "test-org"

    model = {}
    model_name = "gpt-4"
    custom_llm_provider = None
    litellm_params = {
        "api_key": "os.environ/TEST_API_KEY",
        "api_base": "os.environ/TEST_API_BASE",
        "api_version": "os.environ/TEST_API_VERSION",
        "timeout": "os.environ/TEST_TIMEOUT",
        "stream_timeout": "os.environ/TEST_STREAM_TIMEOUT",
        "max_retries": "os.environ/TEST_MAX_RETRIES",
        "organization": "os.environ/TEST_ORGANIZATION",
    }
    default_api_key = None
    default_api_base = None

    result = InitalizeOpenAISDKClient._get_client_initialization_params(
        model=model,
        model_name=model_name,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        default_api_key=default_api_key,
        default_api_base=default_api_base,
    )

    assert isinstance(result, OpenAISDKClientInitializationParams)
    assert result.api_key == "test-api-key"
    assert result.api_base == "https://test.openai.com"
    assert result.api_version == "2023-05-15"
    assert result.timeout == 30.0
    assert result.stream_timeout == 60.0
    assert result.max_retries == 3
    assert result.organization == "test-org"
    assert result.model_name == "gpt-4"
    assert result.custom_llm_provider is None

    # Clean up environment variables
    for key in [
        "TEST_API_KEY",
        "TEST_API_BASE",
        "TEST_API_VERSION",
        "TEST_TIMEOUT",
        "TEST_STREAM_TIMEOUT",
        "TEST_MAX_RETRIES",
        "TEST_ORGANIZATION",
    ]:
        os.environ.pop(key)
