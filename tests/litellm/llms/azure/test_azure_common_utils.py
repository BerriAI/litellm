import json
import os
import sys
import traceback
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.types.utils import CallTypes


# Mock the necessary dependencies
@pytest.fixture
def setup_mocks():
    with patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_entrata_id"
    ) as mock_entrata_token, patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_username_password"
    ) as mock_username_password_token, patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_oidc"
    ) as mock_oidc_token, patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_provider"
    ) as mock_token_provider, patch(
        "litellm.llms.azure.common_utils.litellm"
    ) as mock_litellm, patch(
        "litellm.llms.azure.common_utils.verbose_logger"
    ) as mock_logger, patch(
        "litellm.llms.azure.common_utils.select_azure_base_url_or_endpoint"
    ) as mock_select_url:

        # Configure mocks
        mock_litellm.AZURE_DEFAULT_API_VERSION = "2023-05-15"
        mock_litellm.enable_azure_ad_token_refresh = False

        mock_entrata_token.return_value = lambda: "mock-entrata-token"
        mock_username_password_token.return_value = (
            lambda: "mock-username-password-token"
        )
        mock_oidc_token.return_value = "mock-oidc-token"
        mock_token_provider.return_value = lambda: "mock-default-token"

        mock_select_url.side_effect = (
            lambda azure_client_params, **kwargs: azure_client_params
        )

        yield {
            "entrata_token": mock_entrata_token,
            "username_password_token": mock_username_password_token,
            "oidc_token": mock_oidc_token,
            "token_provider": mock_token_provider,
            "litellm": mock_litellm,
            "logger": mock_logger,
            "select_url": mock_select_url,
        }


def test_initialize_with_api_key(setup_mocks):
    # Test with api_key provided
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
    )

    # Verify expected result
    assert result["api_key"] == "test-api-key"
    assert result["azure_endpoint"] == "https://test.openai.azure.com"
    assert result["api_version"] == "2023-06-01"
    assert "azure_ad_token" in result
    assert result["azure_ad_token"] is None


def test_initialize_with_tenant_credentials(setup_mocks):
    # Test with tenant_id, client_id, and client_secret provided
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={
            "tenant_id": "test-tenant-id",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify that get_azure_ad_token_from_entrata_id was called
    setup_mocks["entrata_token"].assert_called_once_with(
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
    )

    # Verify expected result
    assert result["api_key"] is None
    assert result["azure_endpoint"] == "https://test.openai.azure.com"
    assert "azure_ad_token_provider" in result


def test_initialize_with_username_password(setup_mocks):
    # Test with azure_username, azure_password, and client_id provided
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={
            "azure_username": "test-username",
            "azure_password": "test-password",
            "client_id": "test-client-id",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify that get_azure_ad_token_from_username_password was called
    setup_mocks["username_password_token"].assert_called_once_with(
        azure_username="test-username",
        azure_password="test-password",
        client_id="test-client-id",
    )

    # Verify expected result
    assert "azure_ad_token_provider" in result


def test_initialize_with_oidc_token(setup_mocks):
    # Test with azure_ad_token that starts with "oidc/"
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={"azure_ad_token": "oidc/test-token"},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify that get_azure_ad_token_from_oidc was called
    setup_mocks["oidc_token"].assert_called_once_with("oidc/test-token")

    # Verify expected result
    assert result["azure_ad_token"] == "mock-oidc-token"


def test_initialize_with_enable_token_refresh(setup_mocks):
    # Enable token refresh
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True

    # Test with token refresh enabled
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify that get_azure_ad_token_provider was called
    setup_mocks["token_provider"].assert_called_once()

    # Verify expected result
    assert "azure_ad_token_provider" in result


def test_initialize_with_token_refresh_error(setup_mocks):
    # Enable token refresh but make it raise an error
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True
    setup_mocks["token_provider"].side_effect = ValueError("Token provider error")

    # Test with token refresh enabled but raising error
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
    )

    # Verify error was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Azure AD Token Provider could not be used."
    )


def test_api_version_from_env_var(setup_mocks):
    # Test api_version from environment variable
    with patch.dict(os.environ, {"AZURE_API_VERSION": "2023-07-01"}):
        result = BaseAzureLLM().initialize_azure_sdk_client(
            litellm_params={},
            api_key="test-api-key",
            api_base="https://test.openai.azure.com",
            model_name="gpt-4",
            api_version=None,
        )

    # Verify expected result
    assert result["api_version"] == "2023-07-01"


def test_select_azure_base_url_called(setup_mocks):
    # Test that select_azure_base_url_or_endpoint is called
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
    )

    # Verify that select_azure_base_url_or_endpoint was called
    setup_mocks["select_url"].assert_called_once()


@pytest.mark.parametrize(
    "call_type",
    [
        call_type
        for call_type in CallTypes.__members__.values()
        if call_type.name.startswith("a")
        and call_type.name
        not in [
            "amoderation",
            "arerank",
            "arealtime",
            "anthropic_messages",
            "add_message",
            "arun_thread_stream",
            "aresponses",
        ]
    ],
)
@pytest.mark.asyncio
async def test_ensure_initialize_azure_sdk_client_always_used(call_type):
    from litellm.router import Router

    # Create a router with an Azure model
    azure_model_name = "azure/chatgpt-v-2"
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": azure_model_name,
                    "api_key": "test-api-key",
                    "api_version": os.getenv("AZURE_API_VERSION", "2023-05-15"),
                    "api_base": os.getenv(
                        "AZURE_API_BASE", "https://test.openai.azure.com"
                    ),
                },
            }
        ],
    )

    # Prepare test input based on call type
    test_inputs = {
        "acompletion": {
            "messages": [{"role": "user", "content": "Hello, how are you?"}]
        },
        "atext_completion": {"prompt": "Hello, how are you?"},
        "aimage_generation": {"prompt": "Hello, how are you?"},
        "aembedding": {"input": "Hello, how are you?"},
        "arerank": {"input": "Hello, how are you?"},
        "atranscription": {"file": "path/to/file"},
        "aspeech": {"input": "Hello, how are you?", "voice": "female"},
        "acreate_batch": {
            "completion_window": 10,
            "endpoint": "https://test.openai.azure.com",
            "input_file_id": "123",
        },
        "aretrieve_batch": {"batch_id": "123"},
        "aget_assistants": {"custom_llm_provider": "azure"},
        "acreate_assistants": {"custom_llm_provider": "azure"},
        "adelete_assistant": {"custom_llm_provider": "azure", "assistant_id": "123"},
        "acreate_thread": {"custom_llm_provider": "azure"},
        "aget_thread": {"custom_llm_provider": "azure", "thread_id": "123"},
        "a_add_message": {
            "custom_llm_provider": "azure",
            "thread_id": "123",
            "role": "user",
            "content": "Hello, how are you?",
        },
        "aget_messages": {"custom_llm_provider": "azure", "thread_id": "123"},
        "arun_thread": {
            "custom_llm_provider": "azure",
            "assistant_id": "123",
            "thread_id": "123",
        },
        "acreate_file": {
            "custom_llm_provider": "azure",
            "file": MagicMock(),
            "purpose": "assistants",
        },
    }

    # Get appropriate input for this call type
    input_kwarg = test_inputs.get(call_type.value, {})

    patch_target = "litellm.main.azure_chat_completions.initialize_azure_sdk_client"
    if call_type == CallTypes.atranscription:
        patch_target = (
            "litellm.main.azure_audio_transcriptions.initialize_azure_sdk_client"
        )
    elif call_type == CallTypes.arerank:
        patch_target = (
            "litellm.rerank_api.main.azure_rerank.initialize_azure_sdk_client"
        )
    elif call_type == CallTypes.acreate_batch or call_type == CallTypes.aretrieve_batch:
        patch_target = (
            "litellm.batches.main.azure_batches_instance.initialize_azure_sdk_client"
        )
    elif (
        call_type == CallTypes.aget_assistants
        or call_type == CallTypes.acreate_assistants
        or call_type == CallTypes.adelete_assistant
        or call_type == CallTypes.acreate_thread
        or call_type == CallTypes.aget_thread
        or call_type == CallTypes.a_add_message
        or call_type == CallTypes.aget_messages
        or call_type == CallTypes.arun_thread
    ):
        patch_target = (
            "litellm.assistants.main.azure_assistants_api.initialize_azure_sdk_client"
        )
    elif call_type == CallTypes.acreate_file or call_type == CallTypes.afile_content:
        patch_target = (
            "litellm.files.main.azure_files_instance.initialize_azure_sdk_client"
        )

    # Mock the initialize_azure_sdk_client function
    with patch(patch_target) as mock_init_azure:
        # Also mock async_function_with_fallbacks to prevent actual API calls
        # Call the appropriate router method
        try:
            get_attr = getattr(router, call_type.value, None)
            if get_attr is None:
                pytest.skip(
                    f"Skipping {call_type.value} because it is not supported on Router"
                )
            await getattr(router, call_type.value)(
                model="gpt-3.5-turbo",
                **input_kwarg,
                num_retries=0,
                azure_ad_token="oidc/test-token",
            )
        except Exception as e:
            traceback.print_exc()

        # Verify initialize_azure_sdk_client was called
        mock_init_azure.assert_called_once()

        # Verify it was called with the right model name
        calls = mock_init_azure.call_args_list
        azure_calls = [call for call in calls]

        litellm_params = azure_calls[0].kwargs["litellm_params"]
        print("litellm_params", litellm_params)

        assert (
            "azure_ad_token" in litellm_params
        ), "azure_ad_token not found in parameters"
        assert (
            litellm_params["azure_ad_token"] == "oidc/test-token"
        ), "azure_ad_token is not correct"

        # More detailed verification (optional)
        for call in azure_calls:
            assert "api_key" in call.kwargs, "api_key not found in parameters"
            assert "api_base" in call.kwargs, "api_base not found in parameters"


@pytest.mark.parametrize(
    "call_type",
    [
        CallTypes.atext_completion,
        CallTypes.acompletion,
    ],
)
@pytest.mark.asyncio
async def test_ensure_initialize_azure_sdk_client_always_used_azure_text(call_type):
    from litellm.router import Router

    # Create a router with an Azure model
    azure_model_name = "azure_text/chatgpt-v-2"
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": azure_model_name,
                    "api_key": "test-api-key",
                    "api_version": os.getenv("AZURE_API_VERSION", "2023-05-15"),
                    "api_base": os.getenv(
                        "AZURE_API_BASE", "https://test.openai.azure.com"
                    ),
                },
            }
        ],
    )

    # Prepare test input based on call type
    test_inputs = {
        "acompletion": {
            "messages": [{"role": "user", "content": "Hello, how are you?"}]
        },
        "atext_completion": {"prompt": "Hello, how are you?"},
    }

    # Get appropriate input for this call type
    input_kwarg = test_inputs.get(call_type.value, {})

    patch_target = "litellm.main.azure_text_completions.initialize_azure_sdk_client"

    # Mock the initialize_azure_sdk_client function
    with patch(patch_target) as mock_init_azure:
        # Also mock async_function_with_fallbacks to prevent actual API calls
        # Call the appropriate router method
        try:
            get_attr = getattr(router, call_type.value, None)
            if get_attr is None:
                pytest.skip(
                    f"Skipping {call_type.value} because it is not supported on Router"
                )
            await getattr(router, call_type.value)(
                model="gpt-3.5-turbo",
                **input_kwarg,
                num_retries=0,
                azure_ad_token="oidc/test-token",
            )
        except Exception as e:
            traceback.print_exc()

        # Verify initialize_azure_sdk_client was called
        mock_init_azure.assert_called_once()

        # Verify it was called with the right model name
        calls = mock_init_azure.call_args_list
        azure_calls = [call for call in calls]

        litellm_params = azure_calls[0].kwargs["litellm_params"]
        print("litellm_params", litellm_params)

        assert (
            "azure_ad_token" in litellm_params
        ), "azure_ad_token not found in parameters"
        assert (
            litellm_params["azure_ad_token"] == "oidc/test-token"
        ), "azure_ad_token is not correct"

        # More detailed verification (optional)
        for call in azure_calls:
            assert "api_key" in call.kwargs, "api_key not found in parameters"
            assert "api_base" in call.kwargs, "api_base not found in parameters"
