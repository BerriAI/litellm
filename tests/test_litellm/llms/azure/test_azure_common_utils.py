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
    )

    # Verify expected result
    assert "azure_ad_token_provider" in result


def test_initialize_with_oidc_token(setup_mocks, monkeypatch):
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    # Test with azure_ad_token that starts with "oidc/"
    result = BaseAzureLLM().initialize_azure_sdk_client(
        litellm_params={"azure_ad_token": "oidc/test-token"},
        api_key=None,
        api_base="https://test.openai.azure.com",
        model_name="gpt-4",
        api_version=None,
        is_async=False,
    )

    # Verify that get_azure_ad_token_from_oidc was called
    setup_mocks["oidc_token"].assert_called_once_with(
        azure_ad_token="oidc/test-token", azure_client_id=None, azure_tenant_id=None
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
    )

    # Verify expected result
    assert result["azure_ad_token"] == "mock-oidc-token"


def test_initialize_with_oidc_token_no_credentials(setup_mocks, monkeypatch):
    # Clear environment variables
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)

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
        azure_ad_token="oidc/test-token", azure_client_id=None, azure_tenant_id=None
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
            "acreate_fine_tuning_job",
            "acancel_fine_tuning_job",
            "alist_fine_tuning_jobs",
            "aretrieve_fine_tuning_job",
            "afile_list",
            "aimage_edit",
            "image_edit",
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
