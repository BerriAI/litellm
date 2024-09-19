#### What this tests ####
#    This tests client initialization + reinitialization on the router

import asyncio
import os

#### What this tests ####
#    This tests caching on the router
import sys
import time
import traceback
from typing import Dict
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from openai.lib.azure import OpenAIError

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import APIConnectionError, Router


async def test_router_init():
    """
    1. Initializes clients on the router with 0
    2. Checks if client is still valid
    3. Checks if new client was initialized
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo-0613",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "1234"},
            "tpm": 100000,
            "rpm": 10000,
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-2",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": os.getenv("AZURE_API_VERSION"),
            },
            "tpm": 100000,
            "rpm": 10000,
        },
    ]

    messages = [
        {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
    ]
    client_ttl_time = 2
    router = Router(
        model_list=model_list,
        redis_host=os.environ["REDIS_HOST"],
        redis_password=os.environ["REDIS_PASSWORD"],
        redis_port=os.environ["REDIS_PORT"],
        cache_responses=True,
        timeout=30,
        routing_strategy="simple-shuffle",
        client_ttl=client_ttl_time,
    )
    model = "gpt-3.5-turbo"
    cache_key = f"1234_async_client"
    ## ASSERT IT EXISTS AT THE START ##
    assert router.cache.get_cache(key=cache_key) is not None
    response1 = await router.acompletion(model=model, messages=messages, temperature=1)
    await asyncio.sleep(client_ttl_time)
    ## ASSERT IT'S CLEARED FROM CACHE ##
    assert router.cache.get_cache(key=cache_key, local_only=True) is None
    ## ASSERT IT EXISTS AFTER RUNNING __GET_CLIENT() ##
    assert (
        router._get_client(
            deployment=model_list[0], client_type="async", kwargs={"stream": False}
        )
        is not None
    )


@patch("litellm.secret_managers.get_azure_ad_token_provider.os")
def test_router_init_with_neither_api_key_nor_azure_service_principal_with_secret(
    mocked_os_lib: MagicMock,
) -> None:
    """
    Test router initialization with neither API key nor using Azure Service Principal with Secret authentication
    workflow (having not provided environment variables).
    """
    litellm.enable_azure_ad_token_refresh = True
    # mock EMPTY environment variables
    environment_variables_expected_to_use: Dict = {}
    mocked_environ = PropertyMock(return_value=environment_variables_expected_to_use)
    # Because of the way mock attributes are stored you can’t directly attach a PropertyMock to a mock object.
    # https://docs.python.org/3.11/library/unittest.mock.html#unittest.mock.PropertyMock
    type(mocked_os_lib).environ = mocked_environ

    # define the model list
    model_list = [
        {
            # test case for Azure Service Principal with Secret authentication
            "model_name": "gpt-4o",
            "litellm_params": {
                # checkout there is no api_key here -
                # AZURE_CLIENT_ID, AZURE_CLIENT_SECRET and AZURE_TENANT_ID environment variables should be used instead
                "model": "gpt-4o",
                "base_model": "gpt-4o",
                "api_base": "test_api_base",
                "api_version": "2024-01-01-preview",
                "custom_llm_provider": "azure",
            },
            "model_info": {"mode": "completion"},
        },
    ]

    # initialize the router
    with pytest.raises(OpenAIError):
        # it would raise an error, because environment variables were not provided => azure_ad_token_provider is None
        Router(model_list=model_list)

    # check if the mocked environment variables were reached
    mocked_environ.assert_called()


@patch("azure.identity.get_bearer_token_provider")
@patch("azure.identity.ClientSecretCredential")
@patch("litellm.secret_managers.get_azure_ad_token_provider.os")
def test_router_init_azure_service_principal_with_secret_with_environment_variables(
    mocked_os_lib: MagicMock,
    mocked_credential: MagicMock,
    mocked_get_bearer_token_provider: MagicMock,
) -> None:
    """
    Test router initialization and sample completion using Azure Service Principal with Secret authentication workflow,
    having provided the (mocked) credentials in environment variables and not provided any API key.

    To allow for local testing without real credentials, first must mock Azure SDK authentication functions
    and environment variables.
    """
    litellm.enable_azure_ad_token_refresh = True
    # mock the token provider function
    mocked_func_generating_token = MagicMock(return_value="test_token")
    mocked_get_bearer_token_provider.return_value = mocked_func_generating_token

    # mock the environment variables with mocked credentials
    environment_variables_expected_to_use = {
        "AZURE_CLIENT_ID": "test_client_id",
        "AZURE_CLIENT_SECRET": "test_client_secret",
        "AZURE_TENANT_ID": "test_tenant_id",
    }
    mocked_environ = PropertyMock(return_value=environment_variables_expected_to_use)
    # Because of the way mock attributes are stored you can’t directly attach a PropertyMock to a mock object.
    # https://docs.python.org/3.11/library/unittest.mock.html#unittest.mock.PropertyMock
    type(mocked_os_lib).environ = mocked_environ

    # define the model list
    model_list = [
        {
            # test case for Azure Service Principal with Secret authentication
            "model_name": "gpt-4o",
            "litellm_params": {
                # checkout there is no api_key here -
                # AZURE_CLIENT_ID, AZURE_CLIENT_SECRET and AZURE_TENANT_ID environment variables should be used instead
                "model": "gpt-4o",
                "base_model": "gpt-4o",
                "api_base": "test_api_base",
                "api_version": "2024-01-01-preview",
                "custom_llm_provider": "azure",
            },
            "model_info": {"mode": "completion"},
        },
    ]

    # initialize the router
    router = Router(model_list=model_list)

    # first check if environment variables were used at all
    mocked_environ.assert_called()
    # then check if the client was initialized with the correct environment variables
    mocked_credential.assert_called_with(
        **{
            "client_id": environment_variables_expected_to_use["AZURE_CLIENT_ID"],
            "client_secret": environment_variables_expected_to_use[
                "AZURE_CLIENT_SECRET"
            ],
            "tenant_id": environment_variables_expected_to_use["AZURE_TENANT_ID"],
        }
    )
    # check if the token provider was called at all
    mocked_get_bearer_token_provider.assert_called()
    # then check if the token provider was initialized with the mocked credential
    for call_args in mocked_get_bearer_token_provider.call_args_list:
        assert call_args.args[0] == mocked_credential.return_value
    # however, at this point token should not be fetched yet
    mocked_func_generating_token.assert_not_called()

    # now let's try to make a completion call
    deployment = model_list[0]
    model = deployment["model_name"]
    messages = [
        {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
    ]
    with pytest.raises(APIConnectionError):
        # of course, it will raise an error, because URL is mocked
        router.completion(model=model, messages=messages, temperature=1)  # type: ignore

    # finally verify if the mocked token was used by Azure SDK
    mocked_func_generating_token.assert_called()


# asyncio.run(test_router_init())
