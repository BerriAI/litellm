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
from unittest.mock import ANY


@pytest.mark.skip(
    reason="This test is not relevant to the current codebase. The default Azure AD workflow is used."
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
    monkeypatch,
) -> None:
    """
    Test router initialization and sample completion using Azure Service Principal with Secret authentication workflow,
    having provided the (mocked) credentials in environment variables and not provided any API key.

    To allow for local testing without real credentials, first must mock Azure SDK authentication functions
    and environment variables.
    """
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
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

    # # first check if environment variables were used at all
    # mocked_environ.assert_called()
    # # then check if the client was initialized with the correct environment variables
    # mocked_credential.assert_called_with(
    #     **{
    #         "client_id": environment_variables_expected_to_use["AZURE_CLIENT_ID"],
    #         "client_secret": environment_variables_expected_to_use[
    #             "AZURE_CLIENT_SECRET"
    #         ],
    #         "tenant_id": environment_variables_expected_to_use["AZURE_TENANT_ID"],
    #     }
    # )
    # # check if the token provider was called at all
    # mocked_get_bearer_token_provider.assert_called()
    # # then check if the token provider was initialized with the mocked credential
    # for call_args in mocked_get_bearer_token_provider.call_args_list:
    #     assert call_args.args[0] == mocked_credential.return_value
    # # however, at this point token should not be fetched yet
    # mocked_func_generating_token.assert_not_called()

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


@pytest.mark.asyncio
async def test_audio_speech_router():
    """
    Test that router uses OpenAI/Azure OpenAI Client initialized during init for litellm.aspeech
    """

    from litellm import Router

    litellm.set_verbose = True

    model_list = [
        {
            "model_name": "tts",
            "litellm_params": {
                "model": "azure/azure-tts",
                "api_base": os.getenv("AZURE_SWEDEN_API_BASE"),
                "api_key": os.getenv("AZURE_SWEDEN_API_KEY"),
            },
        },
    ]

    _router = Router(model_list=model_list)

    expected_openai_client = _router._get_client(
        deployment=_router.model_list[0],
        kwargs={},
        client_type="async",
    )

    with patch("litellm.aspeech") as mock_aspeech:
        await _router.aspeech(
            model="tts",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
        )

        print(
            "litellm.aspeech was called with kwargs = ", mock_aspeech.call_args.kwargs
        )

        # Get the actual client that was passed
        client_passed_in_request = mock_aspeech.call_args.kwargs["client"]
        assert client_passed_in_request == expected_openai_client
