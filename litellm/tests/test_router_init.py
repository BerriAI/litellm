# this tests if the router is initialized correctly
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
