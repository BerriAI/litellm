"""Test health check helper functions"""

import struct
import zlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
from litellm.litellm_core_utils.health_check_helpers import (
    IMAGE_EDIT_HEALTH_CHECK_PROMPT,
    HealthCheckHelpers,
)
from litellm.main import ahealth_check
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import LIST_BATCHES_SUPPORTED_PROVIDERS


def _png_chunks(png: bytes, offset: int = 8) -> tuple[tuple[bytes, bytes], ...]:
    if offset >= len(png):
        return ()
    (length,) = struct.unpack(">I", png[offset : offset + 4])
    chunk = (png[offset + 4 : offset + 8], png[offset + 8 : offset + 8 + length])
    return (chunk, *_png_chunks(png, offset + 12 + length))


def _distinct_rgb_colors(png: bytes) -> set[bytes]:
    width = int.from_bytes(png[16:20], "big")
    raw = zlib.decompress(b"".join(data for tag, data in _png_chunks(png) if tag == b"IDAT"))
    row_size = 1 + width * 3
    rows = tuple(raw[i : i + row_size] for i in range(0, len(raw), row_size))
    assert all(row[0] == 0 for row in rows)
    return {bytes(row[i : i + 3]) for row in rows for i in range(1, row_size, 3)}


@pytest.mark.asyncio
async def test_image_edit_health_check_handler_uses_descriptive_prompt_and_multicolor_png():
    model_params = {"model": "openai/gpt-image-1", "api_key": "sk-test"}
    mode_handlers = HealthCheckHelpers.get_mode_handlers(
        model="gpt-image-1",
        custom_llm_provider="openai",
        model_params=model_params,
    )

    assert "image_edit" in mode_handlers

    with patch(  # test-quality-ok: the public health-check path has no dependency injection seam
        "litellm.aimage_edit", new_callable=AsyncMock, return_value={}
    ) as mock_aimage_edit:
        await mode_handlers["image_edit"]()
        await HealthCheckHelpers.get_mode_handlers(
            model="gpt-image-1",
            custom_llm_provider="openai",
            model_params=model_params,
            prompt="test from litellm",
        )["image_edit"]()

    assert mock_aimage_edit.call_count == 2
    for handler_call in mock_aimage_edit.call_args_list:
        assert handler_call.kwargs["model"] == "openai/gpt-image-1"
        assert handler_call.kwargs["prompt"] == IMAGE_EDIT_HEALTH_CHECK_PROMPT
    image = mock_aimage_edit.call_args_list[0].kwargs["image"]
    assert isinstance(image, bytes)
    assert image.startswith(b"\x89PNG")
    assert int.from_bytes(image[16:20], "big") == 512
    assert int.from_bytes(image[20:24], "big") == 512
    assert len(_distinct_rgb_colors(image)) >= 2


@pytest.mark.asyncio
async def test_ahealth_check_image_edit_treats_content_policy_violation_as_healthy():
    moderation_error = litellm.ContentPolicyViolationError(
        message="Your request was rejected as a result of our safety system.",
        model="gpt-image-1",
        llm_provider="openai",
    )
    with patch(  # test-quality-ok: the public health-check path has no dependency injection seam
        "litellm.aimage_edit", new_callable=AsyncMock, side_effect=moderation_error
    ):
        result = await ahealth_check(
            {"model": "gpt-image-1", "api_key": "sk-test"},
            mode="image_edit",
        )

    assert "error" not in result


@pytest.mark.asyncio
async def test_ahealth_check_image_edit_treats_moderation_blocked_code_as_healthy():
    moderation_blocked = litellm.BadRequestError(
        message=(
            '{"error": {"code": "moderation_blocked", "message": "Your request was blocked", '
            '"moderation_stage": "output", "type": "invalid_request_error"}}'
        ),
        model="gpt-image-1",
        llm_provider="openai",
    )
    with patch(  # test-quality-ok: the public health-check path has no dependency injection seam
        "litellm.aimage_edit", new_callable=AsyncMock, side_effect=moderation_blocked
    ):
        result = await ahealth_check(
            {"model": "gpt-image-1", "api_key": "sk-test"},
            mode="image_edit",
        )

    assert "error" not in result


@pytest.mark.asyncio
async def test_ahealth_check_image_edit_still_fails_on_non_moderation_errors():
    auth_error = litellm.AuthenticationError(
        message="Incorrect API key provided",
        llm_provider="openai",
        model="gpt-image-1",
    )
    with patch(  # test-quality-ok: the public health-check path has no dependency injection seam
        "litellm.aimage_edit", new_callable=AsyncMock, side_effect=auth_error
    ):
        result = await ahealth_check(
            {"model": "gpt-image-1", "api_key": "sk-bad"},
            mode="image_edit",
        )

    assert "error" in result


@pytest.mark.asyncio
async def test_ahealth_check_supports_image_edit_mode():
    with patch(  # test-quality-ok: the public health-check path has no dependency injection seam
        "litellm.aimage_edit", new_callable=AsyncMock, return_value={}
    ):
        result = await ahealth_check(
            {"model": "gpt-image-1", "api_key": "sk-test"},
            mode="image_edit",
        )

    assert "error" not in result
    assert "Mode image_edit not supported" not in str(result)


def test_update_model_params_with_health_check_tracking_information():
    """Test _update_model_params_with_health_check_tracking_information adds required tracking info."""
    initial_model_params = {"model": "gpt-3.5-turbo", "api_key": "test_key"}

    with patch(
        "litellm.proxy._types.UserAPIKeyAuth.get_litellm_internal_health_check_user_api_key_auth"
    ) as mock_get_auth:
        mock_auth = MagicMock()
        mock_get_auth.return_value = mock_auth

        with patch(
            "litellm.proxy.litellm_pre_call_utils.LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata"
        ) as mock_add_auth:
            mock_add_auth.return_value = {
                **initial_model_params,
                "litellm_metadata": {
                    "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
                    "user_api_key_auth": mock_auth,
                },
            }

            result = HealthCheckHelpers._update_model_params_with_health_check_tracking_information(
                initial_model_params
            )

            # Verify that litellm_metadata was added
            assert "litellm_metadata" in result
            assert result["litellm_metadata"]["tags"] == [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]

            # Verify the auth setup was called
            mock_add_auth.assert_called_once()
            call_args = mock_add_auth.call_args
            assert call_args[1]["user_api_key_dict"] == mock_auth
            assert call_args[1]["_metadata_variable_name"] == "litellm_metadata"


def test_get_metadata_for_health_check_call():
    """Test _get_metadata_for_health_check_call returns correct metadata structure."""
    result = HealthCheckHelpers._get_metadata_for_health_check_call()

    expected_metadata = {
        "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
    }

    assert result == expected_metadata
    assert isinstance(result["tags"], list)
    assert len(result["tags"]) == 1
    assert result["tags"][0] == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME


def test_get_litellm_internal_health_check_user_api_key_auth():
    """Test get_litellm_internal_health_check_user_api_key_auth returns properly configured UserAPIKeyAuth object."""
    result = UserAPIKeyAuth.get_litellm_internal_health_check_user_api_key_auth()

    # Verify the returned object is of correct type
    assert isinstance(result, UserAPIKeyAuth)

    # Verify all fields are set to the expected constant value
    assert result.api_key == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.team_id == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.key_alias == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.team_alias == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME


@pytest.mark.asyncio
async def test_ahealth_check_failure_masks_raw_request_headers():
    """
    Security test: Verify that when ahealth_check() fails, the raw_request_headers
    in raw_request_typed_dict are properly masked to prevent API key leaks.

    This tests the fix for the security vulnerability where Authorization headers
    were being exposed in health check error responses.
    """
    # Use a model configuration that will fail (invalid endpoint)
    test_api_key = "dapi-test-key-1234567890abcdef"
    test_headers = {
        "Authorization": f"Bearer {test_api_key}",
        "Content-Type": "application/json",
    }

    response = await ahealth_check(
        model_params={
            "model": "databricks/dbrx-instruct",
            "api_base": "https://invalid-endpoint-that-will-fail.com/",
            "api_key": test_api_key,
            "headers": test_headers,
        },
        mode="chat",
    )

    # Should have error and raw_request_typed_dict
    assert "error" in response
    assert "raw_request_typed_dict" in response

    raw_request_dict = response["raw_request_typed_dict"]
    assert raw_request_dict is not None
    assert isinstance(raw_request_dict, dict)
    assert "raw_request_headers" in raw_request_dict

    headers = raw_request_dict["raw_request_headers"]
    assert headers is not None

    # Security check: Authorization header should be masked, not show full key
    if "Authorization" in headers:
        auth_header = headers["Authorization"]
        # Should be masked (e.g., "Be****90" or similar)
        assert auth_header != f"Bearer {test_api_key}", "Authorization header must be masked"
        assert auth_header != test_api_key, "API key must not appear in Authorization header"
        # Masked headers typically have asterisks or are truncated
        assert "*" in auth_header or len(auth_header) < len(f"Bearer {test_api_key}"), (
            f"Authorization header should be masked but got: {auth_header}"
        )

    # Content-Type should remain unmasked (not sensitive)
    if "Content-Type" in headers:
        assert headers["Content-Type"] == "application/json"

    print(f"Masked Authorization header: {headers.get('Authorization', 'NOT FOUND')}")


@pytest.mark.asyncio
async def test_batch_health_check_bridges_metadata_into_logging_obj():
    """_batch_health_check must call update_from_kwargs on the pre-injected
    logging object so callbacks receive identity/tracking fields in
    model_call_details["litellm_params"]["metadata"]."""
    mock_logging_obj = MagicMock()
    mock_logging_obj.update_from_kwargs = MagicMock()

    litellm_metadata = {
        "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
        "user_api_key_alias": "health-check-key",
    }

    filtered_model_params = {
        "model": "openai/gpt-4",
        "api_base": "https://api.openai.com",
        "litellm_logging_obj": mock_logging_obj,
        "litellm_metadata": litellm_metadata,
    }

    with patch("litellm.alist_batches", new_callable=AsyncMock, return_value={}):
        await HealthCheckHelpers._batch_health_check(
            custom_llm_provider="openai",
            model_params={"model": "openai/gpt-4"},
            filtered_model_params=filtered_model_params,
        )

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4"
    assert call_kwargs["kwargs"] is filtered_model_params
    assert call_kwargs["litellm_params"] == {"api_base": "https://api.openai.com"}


@pytest.mark.asyncio
async def test_batch_health_check_omits_api_base_when_absent():
    """api_base must not appear in litellm_params when the provider resolves
    it implicitly (bedrock, vertex, gemini)."""
    mock_logging_obj = MagicMock()
    mock_logging_obj.update_from_kwargs = MagicMock()

    litellm_metadata = {"tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]}

    filtered_model_params = {
        "model": "bedrock/anthropic.claude-v2",
        "litellm_logging_obj": mock_logging_obj,
        "litellm_metadata": litellm_metadata,
    }

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value={}):
        await HealthCheckHelpers._batch_health_check(
            custom_llm_provider="bedrock",
            model_params={"model": "bedrock/anthropic.claude-v2"},
            filtered_model_params=filtered_model_params,
        )

    call_kwargs = mock_logging_obj.update_from_kwargs.call_args[1]
    assert call_kwargs["litellm_params"] is None


@pytest.mark.asyncio
async def test_batch_health_check_skips_bridge_when_no_logging_obj():
    """When litellm_logging_obj is absent, dispatch still proceeds."""
    litellm_metadata = {"tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]}

    filtered_model_params = {
        "model": "openai/gpt-4",
        "litellm_metadata": litellm_metadata,
    }

    with patch("litellm.alist_batches", new_callable=AsyncMock, return_value={}) as mock_alist:
        await HealthCheckHelpers._batch_health_check(
            custom_llm_provider="openai",
            model_params={"model": "openai/gpt-4"},
            filtered_model_params=filtered_model_params,
        )
        mock_alist.assert_called_once()


@pytest.mark.asyncio
async def test_batch_health_check_uses_alist_batches_for_supported_providers():
    """Providers in LIST_BATCHES_SUPPORTED_PROVIDERS dispatch to alist_batches."""
    mock_logging_obj = MagicMock()
    mock_logging_obj.update_from_kwargs = MagicMock()

    litellm_metadata = {"tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]}

    for provider in LIST_BATCHES_SUPPORTED_PROVIDERS:
        filtered_model_params = {
            "model": f"{provider}/some-model",
            "litellm_logging_obj": mock_logging_obj,
            "litellm_metadata": litellm_metadata,
        }

        with patch("litellm.alist_batches", new_callable=AsyncMock, return_value={}) as mock_alist:
            await HealthCheckHelpers._batch_health_check(
                custom_llm_provider=provider,
                model_params={"model": f"{provider}/some-model"},
                filtered_model_params=filtered_model_params,
            )
            mock_alist.assert_called_once()


@pytest.mark.asyncio
async def test_batch_health_check_falls_back_to_acompletion_for_unsupported():
    """Providers not in LIST_BATCHES_SUPPORTED_PROVIDERS fall back to acompletion."""
    mock_logging_obj = MagicMock()
    mock_logging_obj.update_from_kwargs = MagicMock()

    litellm_metadata = {"tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]}

    filtered_model_params = {
        "model": "bedrock/anthropic.claude-v2",
        "litellm_logging_obj": mock_logging_obj,
        "litellm_metadata": litellm_metadata,
    }

    model_params = {"model": "bedrock/anthropic.claude-v2", "messages": []}

    with (
        patch("litellm.alist_batches", new_callable=AsyncMock) as mock_alist,
        patch("litellm.acompletion", new_callable=AsyncMock, return_value={}) as mock_acompletion,
    ):
        await HealthCheckHelpers._batch_health_check(
            custom_llm_provider="bedrock",
            model_params=model_params,
            filtered_model_params=filtered_model_params,
        )
        mock_alist.assert_not_called()
        mock_acompletion.assert_called_once_with(**model_params)


class _FakeWebsocketConnect:
    def __init__(self, calls, url, **kwargs):
        calls.append({"url": url, **kwargs})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_realtime_health_check_uses_model_level_vertex_params():
    """Regression test: realtime health checks must resolve vertex_credentials,
    vertex_project, and vertex_location from the model row's params instead of
    falling back to process-global VERTEXAI_* settings."""
    import litellm
    from litellm.realtime_api import main as realtime_main

    fake_vertex_base = MagicMock()
    fake_vertex_base.get_vertex_region = MagicMock(return_value="us-central1")
    fake_vertex_base._ensure_access_token_async = AsyncMock(return_value=("model-level-token", "model-level-project"))
    connect_calls = []

    with (
        patch.object(realtime_main, "vertex_llm_base", fake_vertex_base),
        patch(
            "websockets.connect",
            lambda url, **kwargs: _FakeWebsocketConnect(connect_calls, url, **kwargs),
        ),
        patch.object(
            HealthCheckHelpers,
            "_update_model_params_with_health_check_tracking_information",
            staticmethod(lambda model_params: model_params),
        ),
    ):
        result = await litellm.ahealth_check(
            model_params={
                "model": "vertex_ai/gemini-live-2.5-flash-native-audio",
                "vertex_credentials": '{"type":"service_account"}',
                "vertex_project": "model-level-project",
                "vertex_location": "us-central1",
            },
            mode="realtime",
        )

    assert result == {}
    fake_vertex_base.get_vertex_region.assert_called_once_with(
        vertex_region="us-central1", model="gemini-live-2.5-flash-native-audio"
    )
    fake_vertex_base._ensure_access_token_async.assert_called_once_with(
        credentials='{"type":"service_account"}',
        project_id="model-level-project",
        custom_llm_provider="vertex_ai",
    )
    assert connect_calls[0]["url"] == (
        "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    )
    assert connect_calls[0]["additional_headers"] == {
        "Authorization": "Bearer model-level-token",
        "x-goog-user-project": "model-level-project",
    }
