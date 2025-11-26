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
from litellm.llms.azure.common_utils import BaseAzureLLM, get_azure_ad_token
from litellm.secret_managers.get_azure_ad_token_provider import (
    get_azure_ad_token_provider,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.secret_managers.get_azure_ad_token_provider import (
    AzureCredentialType,
)
from litellm.types.utils import CallTypes


# Mock the necessary dependencies
@pytest.fixture
def setup_mocks(monkeypatch):
    # Clear Azure environment variables that might interfere with tests
    monkeypatch.delenv("AZURE_USERNAME", raising=False)
    monkeypatch.delenv("AZURE_PASSWORD", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    monkeypatch.delenv("AZURE_SCOPE", raising=False)
    monkeypatch.delenv("AZURE_AD_TOKEN", raising=False)

    with patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id"
    ) as mock_entra_token, patch(
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

        mock_entra_token.return_value = lambda: "mock-entra-token"
        mock_username_password_token.return_value = (
            lambda: "mock-username-password-token"
        )
        mock_oidc_token.return_value = "mock-oidc-token"
        mock_token_provider.return_value = lambda: "mock-default-token"

        mock_select_url.side_effect = (
            lambda azure_client_params, **kwargs: azure_client_params
        )

        yield {
            "entra_token": mock_entra_token,
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
        is_async=False,
    )

    # Verify expected result
    assert result["api_key"] == "test-api-key"
    assert result["azure_endpoint"] == "https://test.openai.azure.com"
    assert result["api_version"] == "2023-06-01"
    assert "azure_ad_token" in result
    assert result["azure_ad_token"] is None


def test_initialize_with_tenant_credentials_env_var(setup_mocks, monkeypatch):
    monkeypatch.setenv("AZURE_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("AZURE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("AZURE_SCOPE", "test-azure-scope")

    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Verify that get_azure_ad_token_from_entra_id was called
    setup_mocks["entra_token"].assert_called_once_with(
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
        scope="test-azure-scope",
    )

    # Verify expected result
    assert result["api_key"] is None
    assert result["azure_endpoint"] == "https://test.openai.azure.com"
    assert "azure_ad_token_provider" in result


def test_initialize_with_tenant_credentials(setup_mocks):
    # Test with tenant_id, client_id, and client_secret provided
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={
            "tenant_id": "test-tenant-id",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "azure_scope": "test-azure-scope",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Verify that get_azure_ad_token_from_entra_id was called
    setup_mocks["entra_token"].assert_called_once_with(
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
        scope="test-azure-scope",
    )

    # Verify expected result
    assert result["api_key"] is None
    assert result["azure_endpoint"] == "https://test.openai.azure.com"
    assert "azure_ad_token_provider" in result


def test_initialize_with_username_password(monkeypatch, setup_mocks):
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("AZURE_USERNAME", raising=False)
    monkeypatch.delenv("AZURE_PASSWORD", raising=False)
    monkeypatch.delenv("AZURE_SCOPE", raising=False)

    # Test with azure_username, azure_password, and client_id provided
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={
            "azure_username": "test-username",
            "azure_password": "test-password",
            "client_id": "test-client-id",
            "azure_scope": "test-azure-scope",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Print the call arguments for debugging
    print("\nDebug - Call arguments for all mocks:")
    print("username_password_token:", setup_mocks["username_password_token"].call_args)
    print("entra_token:", setup_mocks["entra_token"].call_args)
    print("oidc_token:", setup_mocks["oidc_token"].call_args)
    print("token_provider:", setup_mocks["token_provider"].call_args)
    print("\nResult:", result)

    # Verify that get_azure_ad_token_from_username_password was called
    setup_mocks["username_password_token"].assert_called_once_with(
        azure_username="test-username",
        azure_password="test-password",
        client_id="test-client-id",
        scope="test-azure-scope",
    )

    # Verify expected result
    assert "azure_ad_token_provider" in result


def test_initialize_with_oidc_token(setup_mocks, monkeypatch):
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    monkeypatch.delenv("AZURE_SCOPE", raising=False)

    # Test with azure_ad_token that starts with "oidc/"
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={"azure_ad_token": "oidc/test-token"},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    setup_mocks["oidc_token"].assert_called_once_with(
        azure_ad_token="oidc/test-token",
        azure_client_id=None,
        azure_tenant_id=None,
        scope="https://cognitiveservices.azure.com/.default",
    )

    # Verify expected result
    assert result["azure_ad_token"] == "mock-oidc-token"


def test_initialize_with_oidc_token_and_client_params(setup_mocks):
    # Test with azure_ad_token that starts with "oidc/" and explicit client/tenant IDs
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={
            "azure_ad_token": "oidc/test-token",
            "client_id": "test-client-id",
            "tenant_id": "test-tenant-id",
            "azure_scope": "test-azure-scope",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Verify that get_azure_ad_token_from_oidc was called with the correct parameters
    setup_mocks["oidc_token"].assert_called_once_with(
        azure_ad_token="oidc/test-token",
        azure_client_id="test-client-id",
        azure_tenant_id="test-tenant-id",
        scope="test-azure-scope",
    )

    # Verify expected result
    assert result["azure_ad_token"] == "mock-oidc-token"


def test_initialize_with_oidc_token_fallback_to_env(setup_mocks, monkeypatch):
    # Set environment variables
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("AZURE_TENANT_ID", "env-tenant-id")

    # Test with azure_ad_token that starts with "oidc/" but no explicit client/tenant IDs
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={
            "azure_ad_token": "oidc/test-token",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Verify that get_azure_ad_token_from_oidc was called with environment variables
    setup_mocks["oidc_token"].assert_called_once_with(
        azure_ad_token="oidc/test-token",
        azure_client_id="env-client-id",
        azure_tenant_id="env-tenant-id",
        scope="https://cognitiveservices.azure.com/.default",
    )

    # Verify expected result
    assert result["azure_ad_token"] == "mock-oidc-token"


def test_initialize_with_oidc_token_no_credentials(setup_mocks, monkeypatch):
    # Clear environment variables
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    monkeypatch.delenv("AZURE_SCOPE", raising=False)

    # Test with azure_ad_token that starts with "oidc/" but no credentials anywhere
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={
            "azure_ad_token": "oidc/test-token",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Verify that get_azure_ad_token_from_oidc was called with None values
    setup_mocks["oidc_token"].assert_called_once_with(
        azure_ad_token="oidc/test-token",
        azure_client_id=None,
        azure_tenant_id=None,
        scope="https://cognitiveservices.azure.com/.default",
    )

    # Verify expected result
    assert result["azure_ad_token"] == "mock-oidc-token"


def test_initialize_with_ad_token_provider(setup_mocks, monkeypatch):
    # Clear environment variables
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)

    # Test with custom azure_ad_token_provider
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={
            "azure_ad_token_provider": lambda: "mock-custom-token",
        },
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Verify expected result
    assert result["azure_ad_token_provider"]() == "mock-custom-token"


def test_initialize_with_enable_token_refresh(setup_mocks, monkeypatch):
    litellm._turn_on_debug()
    # Enable token refresh
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True

    # Test with token refresh enabled
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Verify that get_azure_ad_token_provider was called
    setup_mocks["token_provider"].assert_called_once()

    # Verify expected result
    assert "azure_ad_token_provider" in result


def test_initialize_with_token_refresh_error(setup_mocks, monkeypatch):
    # Enable token refresh but make it raise an error
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True
    setup_mocks["token_provider"].side_effect = ValueError("Token provider error")

    # Test with token refresh enabled but raising error
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
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
            is_async=False,
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
        is_async=False,
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
            "alist_input_items",
            "acreate_fine_tuning_job",
            "acancel_fine_tuning_job",
            "alist_fine_tuning_jobs",
            "aretrieve_fine_tuning_job",
            "afile_list",
            "aimage_edit",
            "image_edit",
            "agenerate_content_stream",
            "agenerate_content",
            "allm_passthrough_route",
            "llm_passthrough_route",
            "asearch",
            "avector_store_create",
            "avector_store_search",
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
        "afile_content": {
            "custom_llm_provider": "azure",
            "file_id": "123",
        },
        "afile_delete": {
            "custom_llm_provider": "azure",
            "file_id": "123",
        },
        "avideo_content": {
            "custom_llm_provider": "azure",
            "video_id": "123",
        },
        "avideo_list": {
            "custom_llm_provider": "azure",
        },
        "avideo_remix": {
            "custom_llm_provider": "azure",
            "video_id": "123",
            "prompt": "A new video based on this one",
        },
    }

    # Get appropriate input for this call type
    input_kwarg = test_inputs.get(call_type.value, {})

    patch_target = (
        "litellm.llms.azure.common_utils.BaseAzureLLM.initialize_azure_sdk_client"
    )
    if call_type == CallTypes.arerank:
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
    elif (
        call_type == CallTypes.avideo_content
        or call_type == CallTypes.avideo_list
        or call_type == CallTypes.avideo_remix
    ):
        # Skip video call types as they don't use Azure SDK client initialization
        pytest.skip(f"Skipping {call_type.value} because Azure video calls don't use initialize_azure_sdk_client")
    elif (
        call_type == CallTypes.alist_containers
        or call_type == CallTypes.aretrieve_container
        or call_type == CallTypes.acreate_container
        or call_type == CallTypes.adelete_container
    ):
        # Skip container call types as they're not supported for Azure (only OpenAI)
        pytest.skip(f"Skipping {call_type.value} because Azure doesn't support container operations")
    elif call_type == CallTypes.avector_store_file_create or call_type == CallTypes.avector_store_file_list or call_type == CallTypes.avector_store_file_retrieve or call_type == CallTypes.avector_store_file_content or call_type == CallTypes.avector_store_file_update or call_type == CallTypes.avector_store_file_delete:
        # Skip vector store file call types as they're not supported for Azure (only OpenAI)
        pytest.skip(f"Skipping {call_type.value} because Azure doesn't support vector store file operations")
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


# Test parameters for different API functions with Azure models
AZURE_API_FUNCTION_PARAMS = [
    # (function_name, is_async, args)
    (
        "completion",
        False,
        {
            "model": "azure/gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "completion",
        True,
        {
            "model": "azure/gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "embedding",
        False,
        {
            "model": "azure/text-embedding-ada-002",
            "input": "Hello world",
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "embedding",
        True,
        {
            "model": "azure/text-embedding-ada-002",
            "input": "Hello world",
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "speech",
        False,
        {
            "model": "azure/tts-1",
            "input": "Hello, this is a test of text to speech",
            "voice": "alloy",
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "speech",
        True,
        {
            "model": "azure/tts-1",
            "input": "Hello, this is a test of text to speech",
            "voice": "alloy",
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "transcription",
        False,
        {
            "model": "azure/whisper-1",
            "file": MagicMock(),
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "transcription",
        True,
        {
            "model": "azure/whisper-1",
            "file": MagicMock(),
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
]


@pytest.mark.parametrize("function_name,is_async,args", AZURE_API_FUNCTION_PARAMS)
@pytest.mark.asyncio
async def test_azure_client_reuse(function_name, is_async, args):
    """
    Test that multiple Azure API calls reuse the same Azure OpenAI client
    """
    litellm.set_verbose = True

    # Determine which client class to mock based on whether the test is async
    client_path = (
        "litellm.llms.azure.common_utils.AsyncAzureOpenAI"
        if is_async
        else "litellm.llms.azure.common_utils.AzureOpenAI"
    )

    # Create a proper mock class that can pass isinstance checks
    mock_client = MagicMock()

    # Create the appropriate patches
    with patch(client_path) as mock_client_class, patch.object(
        BaseAzureLLM, "set_cached_openai_client"
    ) as mock_set_cache, patch.object(
        BaseAzureLLM, "get_cached_openai_client"
    ) as mock_get_cache, patch.object(
        BaseAzureLLM, "initialize_azure_sdk_client"
    ) as mock_init_azure:
        # Configure the mock client class to return our mock instance
        mock_client_class.return_value = mock_client

        # Setup the mock to return None first time (cache miss) then a client for subsequent calls
        mock_get_cache.side_effect = [None] + [
            mock_client
        ] * 9  # First call returns None, rest return the mock client

        # Mock the initialize_azure_sdk_client to return a dict with the necessary params
        mock_init_azure.return_value = {
            "api_key": args.get("api_key"),
            "azure_endpoint": args.get("api_base"),
            "api_version": args.get("api_version"),
            "azure_ad_token": None,
            "azure_ad_token_provider": None,
        }

        # Make 10 API calls
        for _ in range(10):
            try:
                # Call the appropriate function based on parameters
                if is_async:
                    # Add 'a' prefix for async functions
                    func = getattr(litellm, f"a{function_name}")
                    await func(**args)
                else:
                    func = getattr(litellm, function_name)
                    func(**args)
            except Exception:
                # We expect exceptions since we're mocking the client
                pass

        # Verify client was created only once
        assert (
            mock_client_class.call_count == 1
        ), f"{'Async' if is_async else ''}AzureOpenAI client should be created only once"

        # Verify initialize_azure_sdk_client was called once
        assert (
            mock_init_azure.call_count == 1
        ), "initialize_azure_sdk_client should be called once"

        # Verify the client was cached
        assert mock_set_cache.call_count == 1, "Client should be cached once"

        # Verify we tried to get from cache 10 times (once per request)
        assert mock_get_cache.call_count == 10, "Should check cache for each request"


@pytest.mark.asyncio
async def test_azure_client_cache_separates_sync_and_async():
    """
    Test that the Azure client cache correctly separates sync and async clients.
    This directly tests the fix for issues #9801 and #10318 where sync and async
    clients were being mixed up in the cache.
    """
    from litellm.llms.azure.common_utils import BaseAzureLLM

    # Clear the in-memory cache before test
    litellm.in_memory_llm_clients_cache._cache = {}

    # Create mock sync and async clients
    mock_sync_client = MagicMock()
    mock_async_client = MagicMock()

    # Patch the Azure client classes
    with patch(
        "litellm.llms.azure.common_utils.AzureOpenAI"
    ) as mock_sync_client_class, patch(
        "litellm.llms.azure.common_utils.AsyncAzureOpenAI"
    ) as mock_async_client_class, patch.object(
        BaseAzureLLM, "initialize_azure_sdk_client"
    ) as mock_init_azure:
        # Configure the mocks to return our instances
        mock_sync_client_class.return_value = mock_sync_client
        mock_async_client_class.return_value = mock_async_client

        # Mock the initialize_azure_sdk_client to return necessary params
        mock_init_azure.return_value = {
            "api_key": "test-api-key",
            "azure_endpoint": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
            "azure_ad_token": None,
            "azure_ad_token_provider": None,
        }

        # Create an instance and make identical requests with different async flags
        base_llm = BaseAzureLLM()
        common_params = {
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
            "model": "gpt-4",
            "litellm_params": {},
        }

        # Get a sync client
        sync_client = base_llm.get_azure_openai_client(_is_async=False, **common_params)
        # Then get an async client with identical parameters
        async_client = base_llm.get_azure_openai_client(_is_async=True, **common_params)

        # Verify we got the right classes
        assert (
            sync_client is mock_sync_client
        ), "Sync client should be the mock sync client"
        assert (
            async_client is mock_async_client
        ), "Async client should be the mock async client"

        # Verify each client class was instantiated exactly once
        assert (
            mock_sync_client_class.call_count == 1
        ), "AzureOpenAI should be instantiated once"
        assert (
            mock_async_client_class.call_count == 1
        ), "AsyncAzureOpenAI should be instantiated once"

        # Verify initialize_azure_sdk_client was called for each client type
        assert (
            mock_init_azure.call_count == 2
        ), "initialize_azure_sdk_client should be called twice"


def test_scope_always_string_in_initialize_azure_sdk_client(setup_mocks, monkeypatch):
    """
    Test that the scope parameter in initialize_azure_sdk_client is always a string,
    regardless of the input provided (None, empty string, etc.).
    """
    # Clear environment variables to ensure clean test state
    monkeypatch.delenv("AZURE_SCOPE", raising=False)

    base_llm = BaseAzureLLM()
    expected_default_scope = "https://cognitiveservices.azure.com/.default"

    # Test case 1: scope is None in litellm_params
    result = base_llm.initialize_azure_sdk_client(
        litellm_params={"azure_scope": None},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
        is_async=False,
    )

    # Verify scope is a string and has the expected default value
    # We need to check the internal logic by inspecting what was passed to mocked functions
    setup_mocks["select_url"].assert_called()
    call_args = setup_mocks["select_url"].call_args[1]["azure_client_params"]
    # The scope should be used internally when setting up token providers

    # Test case 2: azure_scope key is missing entirely
    result = base_llm.initialize_azure_sdk_client(
        litellm_params={},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
        is_async=False,
    )

    # Test case 3: azure_scope is an empty string
    result = base_llm.initialize_azure_sdk_client(
        litellm_params={"azure_scope": ""},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
        is_async=False,
    )

    # Test case 4: azure_scope is a valid custom string
    custom_scope = "https://custom.scope.com/.default"
    result = base_llm.initialize_azure_sdk_client(
        litellm_params={"azure_scope": custom_scope},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
        is_async=False,
    )

    # Test case 5: Test with token authentication to verify scope is passed correctly
    setup_mocks["entra_token"].reset_mock()
    result = base_llm.initialize_azure_sdk_client(
        litellm_params={
            "azure_scope": None,  # This should default to the expected scope
            "tenant_id": "test-tenant",
            "client_id": "test-client",
            "client_secret": "test-secret",
        },
        api_key=None,  # No API key to trigger token authentication
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
        is_async=False,
    )

    # Verify that the token function was called with a string scope
    setup_mocks["entra_token"].assert_called_once()
    call_args = setup_mocks["entra_token"].call_args
    scope_arg = call_args[1]["scope"]  # scope should be passed as keyword argument
    assert isinstance(
        scope_arg, str
    ), f"Scope should be a string, got {type(scope_arg)}"
    assert (
        scope_arg == expected_default_scope
    ), f"Scope should be {expected_default_scope}, got {scope_arg}"

    # Test case 6: Test with environment variable set to None (edge case)
    monkeypatch.setenv("AZURE_SCOPE", "")
    result = base_llm.initialize_azure_sdk_client(
        litellm_params={"azure_scope": None},
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version="2023-06-01",
        is_async=False,
    )

    print("All scope tests passed - scope is always a string")


def test_with_existing_token_provider(setup_mocks):
    """Test get_azure_ad_token with an existing token provider."""
    token_provider = lambda: "test-token"
    litellm_params = GenericLiteLLMParams(azure_ad_token_provider=token_provider)

    token = get_azure_ad_token(litellm_params)

    assert token == "test-token"


def test_with_existing_azure_ad_token(setup_mocks):
    """Test get_azure_ad_token with an existing azure ad token."""
    litellm_params = GenericLiteLLMParams(azure_ad_token="test-token")

    token = get_azure_ad_token(litellm_params)

    assert token == "test-token"


def test_with_existing_azure_ad_token_from_env(setup_mocks):
    """Test get_azure_ad_token with an existing AZURE_AD_TOKEN from env."""

    # mock get_secret_str("AZURE_AD_TOKEN") to "test-token"
    with patch("litellm.llms.azure.common_utils.get_secret_str") as mock_get_secret_str:
        # Configure the mock to return "test-token" when called with "AZURE_AD_TOKEN"
        mock_get_secret_str.side_effect = lambda key: (
            "test-token" if key == "AZURE_AD_TOKEN" else None
        )

        litellm_params = GenericLiteLLMParams()

        token = get_azure_ad_token(litellm_params)

        assert token == "test-token"
        # Verify that get_secret_str was called with "AZURE_AD_TOKEN"
        mock_get_secret_str.assert_called_with("AZURE_AD_TOKEN")


def test_get_azure_ad_token_with_client_id_and_client_secret(setup_mocks):
    """Test get_azure_ad_token with tenant_id, client_id, and client_secret."""
    # Reset mocks to ensure clean state
    setup_mocks["entra_token"].reset_mock()

    # Create test parameters with username, password, and client_id
    # but no other authentication methods
    litellm_params = GenericLiteLLMParams(
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
        azure_scope="test-azure-scope",
    )

    # Call the function
    token = get_azure_ad_token(litellm_params)

    # Verify the debug message was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Using Azure AD Token Provider from Entra ID for Azure Auth"
    )

    # Verify get_azure_ad_token_from_entra_id was called with correct params
    setup_mocks["entra_token"].assert_called_once_with(
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
        scope="test-azure-scope",
    )

    # Verify the token is what we expect from our mock
    assert token == "mock-entra-token"


def test_get_azure_ad_token_with_client_id_and_client_secret_from_env(
    setup_mocks, monkeypatch
):
    """Test get_azure_ad_token with tenant_id, client_id, and client_secret from env."""
    # Reset mocks to ensure clean state
    setup_mocks["entra_token"].reset_mock()

    # Set environment variables
    monkeypatch.setenv("AZURE_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("AZURE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("AZURE_SCOPE", "test-azure-scope")

    # Create test parameters with username, password, and client_id
    # but no other authentication methods
    litellm_params = GenericLiteLLMParams()

    # Call the function
    token = get_azure_ad_token(litellm_params)

    # Verify the debug message was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Using Azure AD Token Provider from Entra ID for Azure Auth"
    )

    # Verify get_azure_ad_token_from_entra_id was called with correct params
    setup_mocks["entra_token"].assert_called_once_with(
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
        scope="test-azure-scope",
    )

    # Verify the token is what we expect from our mock
    assert token == "mock-entra-token"


def test_get_azure_ad_token_with_username_password(setup_mocks):
    """Test get_azure_ad_token with username, password, and client_id."""
    # Reset mocks to ensure clean state
    setup_mocks["username_password_token"].reset_mock()

    # Create test parameters with username, password, and client_id
    # but no other authentication methods
    litellm_params = GenericLiteLLMParams(
        azure_username="test-username",
        azure_password="test-password",
        client_id="test-client-id",
        azure_scope="test-azure-scope",
        # Ensure no other auth methods are available
        azure_ad_token_provider=None,
        azure_ad_token=None,
        tenant_id=None,
        client_secret=None,
    )

    # Call the function
    token = get_azure_ad_token(litellm_params)

    # Verify the debug message was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Using Azure Username and Password for Azure Auth"
    )

    # Verify get_azure_ad_token_from_username_password was called with correct params
    setup_mocks["username_password_token"].assert_called_once_with(
        azure_username="test-username",
        azure_password="test-password",
        client_id="test-client-id",
        scope="test-azure-scope",
    )

    # Verify the token is what we expect from our mock
    assert token == "mock-username-password-token"


def test_get_azure_ad_token_with_missing_username_password(setup_mocks):
    """Test get_azure_ad_token skips username/password auth when credentials are incomplete."""
    # Reset mocks to ensure clean state
    setup_mocks["username_password_token"].reset_mock()

    # Test cases with missing credentials
    test_cases = [
        # Missing username
        GenericLiteLLMParams(
            azure_username=None,
            azure_password="test-password",
            client_id="test-client-id",
        ),
        # Missing password
        GenericLiteLLMParams(
            azure_username="test-username",
            azure_password=None,
            client_id="test-client-id",
        ),
        # Missing client_id
        GenericLiteLLMParams(
            azure_username="test-username",
            azure_password="test-password",
            client_id=None,
        ),
    ]

    for params in test_cases:
        # Call the function
        get_azure_ad_token(params)

        # Verify username/password auth was not used
        setup_mocks["username_password_token"].assert_not_called()

        # Reset mock for next test case
        setup_mocks["username_password_token"].reset_mock()


def test_get_azure_ad_token_with_username_password_from_env(setup_mocks, monkeypatch):
    """Test get_azure_ad_token with username, password, and client_id from environment variables."""
    # Reset mocks to ensure clean state
    setup_mocks["username_password_token"].reset_mock()

    # Set environment variables
    monkeypatch.setenv("AZURE_USERNAME", "env-username")
    monkeypatch.setenv("AZURE_PASSWORD", "env-password")
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("AZURE_SCOPE", "test-azure-scope")

    # Create test parameters with no explicit credentials
    litellm_params = GenericLiteLLMParams(
        # Ensure no other auth methods are available
        azure_ad_token_provider=None,
        azure_ad_token=None,
        tenant_id=None,
        client_secret=None,
        # Don't set username, password, or client_id directly
    )

    # Call the function
    token = get_azure_ad_token(litellm_params)

    # Verify the debug message was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Using Azure Username and Password for Azure Auth"
    )

    # Verify get_azure_ad_token_from_username_password was called with correct params from env
    setup_mocks["username_password_token"].assert_called_once_with(
        azure_username="env-username",
        azure_password="env-password",
        client_id="env-client-id",
        scope="test-azure-scope",
    )

    # Verify the token is what we expect from our mock
    assert token == "mock-username-password-token"


def test_get_azure_ad_token_with_oidc_token(setup_mocks, monkeypatch):
    """Test get_azure_ad_token with OIDC token."""
    # Reset mocks to ensure clean state
    setup_mocks["oidc_token"].reset_mock()

    # Clear environment variables that might interfere with OIDC token logic
    monkeypatch.delenv("AZURE_USERNAME", raising=False)
    monkeypatch.delenv("AZURE_PASSWORD", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)

    # Create test parameters with OIDC token, client_id, and tenant_id
    litellm_params = GenericLiteLLMParams(
        azure_ad_token="oidc/test-token",
        client_id="test-client-id",
        tenant_id="test-tenant-id",
        azure_scope="test-azure-scope",
        # Ensure no other auth methods are available
        azure_ad_token_provider=None,
        client_secret=None,
        azure_username=None,
        azure_password=None,
    )

    # Call the function
    token = get_azure_ad_token(litellm_params)

    # Verify the debug message was logged
    setup_mocks["logger"].debug.assert_any_call("Using Azure OIDC Token for Azure Auth")

    # Verify get_azure_ad_token_from_oidc was called with correct params
    setup_mocks["oidc_token"].assert_called_once_with(
        azure_ad_token="oidc/test-token",
        azure_client_id="test-client-id",
        azure_tenant_id="test-tenant-id",
        scope="test-azure-scope",
    )

    # Verify the token is what we expect from our mock
    assert token == "mock-oidc-token"


def test_get_azure_ad_token_with_token_refresh(setup_mocks, monkeypatch):
    """Test get_azure_ad_token with token refresh enabled."""
    # Reset mocks to ensure clean state
    monkeypatch.delenv("AZURE_USERNAME", raising=False)
    monkeypatch.delenv("AZURE_PASSWORD", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)

    setup_mocks["token_provider"].reset_mock()

    # Enable token refresh
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True

    # Create test parameters with no other auth methods available
    litellm_params = GenericLiteLLMParams()

    # Call the function
    token = get_azure_ad_token(litellm_params)

    # Verify the debug message was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Using Azure AD token provider based on Service Principal with Secret workflow or DefaultAzureCredential for Azure Auth"
    )

    # Verify get_azure_ad_token_provider was called
    setup_mocks["token_provider"].assert_called_once()

    # Verify the token is what we expect from our mock
    assert token == "mock-default-token"


def test_get_azure_ad_token_with_token_refresh_error(setup_mocks):
    """Test get_azure_ad_token with token refresh enabled but raising an error."""
    # Reset mocks to ensure clean state
    setup_mocks["token_provider"].reset_mock()

    # Enable token refresh but make it raise an error
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True
    setup_mocks["token_provider"].side_effect = ValueError("Token provider error")

    # Create test parameters with no other auth methods available
    litellm_params = GenericLiteLLMParams()

    # Call the function
    token = get_azure_ad_token(litellm_params)

    # Verify the debug message was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Using Azure AD token provider based on Service Principal with Secret workflow or DefaultAzureCredential for Azure Auth"
    )

    # Verify error was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Azure AD Token Provider could not be used."
    )

    # Verify get_azure_ad_token_provider was called twice (once for service principal, once for DefaultAzureCredential)
    assert setup_mocks["token_provider"].call_count == 2

    # Verify the token is None since the provider raised an error
    assert token is None


def test_token_provider_returns_non_string(setup_mocks):
    """Test that get_azure_ad_token raises TypeError when token provider returns non-string value."""
    # Create a token provider that returns a non-string value
    non_string_provider = lambda: 123  # Returns an integer instead of a string

    # Create test parameters with the non-string token provider
    litellm_params = GenericLiteLLMParams(azure_ad_token_provider=non_string_provider)

    # Call the function and expect a TypeError
    with pytest.raises(TypeError) as excinfo:
        get_azure_ad_token(litellm_params)

    # Verify the error message
    assert "Azure AD token must be a string" in str(excinfo.value)

    # Verify the error was logged
    setup_mocks["logger"].error.assert_any_call(
        "Azure AD token provider returned non-string value: <class 'int'>"
    )


def test_token_provider_raises_exception(setup_mocks):
    """Test that get_azure_ad_token raises RuntimeError when token provider raises an exception."""
    # Create a token provider that raises an exception
    error_message = "Test provider error"
    error_provider = lambda: exec('raise ValueError("' + error_message + '")')

    # Create test parameters with the error-raising token provider
    litellm_params = GenericLiteLLMParams(azure_ad_token_provider=error_provider)

    # Call the function and expect a RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        get_azure_ad_token(litellm_params)

    # Verify the error message
    assert "Failed to get Azure AD token" in str(excinfo.value)
    assert error_message in str(excinfo.value)

    # Verify the error was logged
    setup_mocks["logger"].error.assert_called()


def test_get_azure_ad_token_provider_with_default_azure_credential():
    """
    Test that get_azure_ad_token_provider correctly uses DefaultAzureCredential 
    when explicitly specified as the credential type. This verifies that the function
    can dynamically instantiate DefaultAzureCredential and return a working token provider.
    """
    # Mock Azure identity classes
    with patch('azure.identity.DefaultAzureCredential') as mock_default_cred, \
         patch('azure.identity.get_bearer_token_provider') as mock_token_provider:
        
        # Configure mocks
        mock_credential_instance = MagicMock()
        mock_default_cred.return_value = mock_credential_instance
        mock_token_provider.return_value = lambda: "test-default-azure-token"
        
        # Test with DefaultAzureCredential specified explicitly
        token_provider = get_azure_ad_token_provider(
            azure_scope="https://cognitiveservices.azure.com/.default",
            azure_credential=AzureCredentialType.DefaultAzureCredential
        )
        
        # Verify DefaultAzureCredential was instantiated
        mock_default_cred.assert_called_once_with()
        
        # Verify get_bearer_token_provider was called with the right parameters
        mock_token_provider.assert_called_once_with(
            mock_credential_instance, 
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Verify the returned token provider works
        token = token_provider()
        assert token == "test-default-azure-token"


def test_get_azure_ad_token_fallback_to_default_azure_credential(setup_mocks, monkeypatch):
    """
    Test that get_azure_ad_token falls back to DefaultAzureCredential when the 
    service principal method fails but token refresh is enabled. This tests the 
    complete fallback flow from service principal to DefaultAzureCredential.
    """
    # Clear environment variables that might interfere
    monkeypatch.delenv("AZURE_USERNAME", raising=False)
    monkeypatch.delenv("AZURE_PASSWORD", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)

    # Reset mocks to ensure clean state
    setup_mocks["token_provider"].reset_mock()

    # Enable token refresh
    setup_mocks["litellm"].enable_azure_ad_token_refresh = True

    # Configure get_azure_ad_token_provider to fail first (service principal) 
    # but succeed on second call (DefaultAzureCredential)
    def mock_token_provider_side_effect(*args, **kwargs):
        # If called with azure_credential=DefaultAzureCredential, return a working provider
        if kwargs.get("azure_credential") == AzureCredentialType.DefaultAzureCredential:
            return lambda: "mock-default-azure-credential-token"
        # Otherwise (service principal call), return None to simulate failure
        return None

    setup_mocks["token_provider"].side_effect = mock_token_provider_side_effect

    # Create test parameters with no other auth methods available
    litellm_params = GenericLiteLLMParams()

    # Call the function
    token = get_azure_ad_token(litellm_params)

    # Verify the success debug message was logged
    setup_mocks["logger"].debug.assert_any_call(
        "Successfully obtained Azure AD token provider using DefaultAzureCredential"
    )

    # Verify get_azure_ad_token_provider was called twice:
    # 1. First with just azure_scope (service principal attempt)
    # 2. Second with azure_credential=DefaultAzureCredential (fallback)
    assert setup_mocks["token_provider"].call_count == 2
    
    # Verify the calls were made with expected parameters
    calls = setup_mocks["token_provider"].call_args_list
    
    # First call should be service principal attempt (no azure_credential)
    first_call_kwargs = calls[0][1]
    assert "azure_scope" in first_call_kwargs
    assert first_call_kwargs.get("azure_credential") is None
    
    # Second call should be DefaultAzureCredential attempt
    second_call_kwargs = calls[1][1]
    assert "azure_scope" in second_call_kwargs
    assert second_call_kwargs.get("azure_credential") == AzureCredentialType.DefaultAzureCredential

    # Verify the token is what we expect from our DefaultAzureCredential mock
    assert token == "mock-default-azure-credential-token"


@pytest.mark.parametrize(
    "api_version,expected",
    [
        ("preview", True),
        ("latest", True),
        ("v1", True),
        (None, False),
        ("2023-05-15", False),
        ("2024-01-01", False),
        ("", False),
    ],
)
def test_is_azure_v1_api_version(api_version, expected):
    """
    Test that _is_azure_v1_api_version correctly identifies v1 API versions.
    """
    result = BaseAzureLLM._is_azure_v1_api_version(api_version=api_version)
    assert result == expected
