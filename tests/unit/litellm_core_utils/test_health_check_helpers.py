"""Test health check helper functions"""

import json
import socket
import struct
import zlib
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx

import litellm
from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
from litellm.litellm_core_utils.health_check_helpers import (
    IMAGE_EDIT_HEALTH_CHECK_PROMPT,
    HealthCheckHelpers,
    default_health_check_mode,
    native_health_check_mode,
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
    """Test update_model_params_with_health_check_tracking_information adds required tracking info."""
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

            result = HealthCheckHelpers.update_model_params_with_health_check_tracking_information(
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
    test_api_key = "dapi-test-key-1234567890abcdef"
    test_headers = {
        "Authorization": f"Bearer {test_api_key}",
        "Content-Type": "application/json",
    }

    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        api_base = f"http://127.0.0.1:{reserved.getsockname()[1]}/"

        response = await ahealth_check(
            model_params={
                "model": "databricks/dbrx-instruct",
                "api_base": api_base,
                "api_key": test_api_key,
                "headers": test_headers,
            },
            mode="chat",
        )

    assert "error" in response
    assert "raw_request_typed_dict" in response

    raw_request_dict = response["raw_request_typed_dict"]
    assert raw_request_dict is not None
    assert isinstance(raw_request_dict, dict)
    assert "raw_request_headers" in raw_request_dict

    headers = raw_request_dict["raw_request_headers"]
    assert headers is not None

    assert "Authorization" in headers
    auth_header = headers["Authorization"]
    assert auth_header != f"Bearer {test_api_key}", "Authorization header must be masked"
    assert auth_header != test_api_key, "API key must not appear in Authorization header"
    assert "*" in auth_header or len(auth_header) < len(f"Bearer {test_api_key}"), (
        f"Authorization header should be masked but got: {auth_header}"
    )

    assert headers["Content-Type"] == "application/json"


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
async def test_batch_health_check_hands_the_resolved_provider_to_alist_batches():
    filtered_model_params: Final = {
        "model": "xai/grok-4.3",
        "api_key": "sk-test",
        "litellm_metadata": {"tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]},
    }

    with patch("litellm.alist_batches", new_callable=AsyncMock, return_value={}) as mock_alist:
        await HealthCheckHelpers._batch_health_check(
            custom_llm_provider="xai",
            model_params={**filtered_model_params, "messages": []},
            filtered_model_params=filtered_model_params,
        )

    assert mock_alist.call_args.kwargs["custom_llm_provider"] == "xai"
    assert mock_alist.call_args.kwargs["model"] == "xai/grok-4.3"
    assert mock_alist.call_args.kwargs["api_key"] == "sk-test"


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
            "update_model_params_with_health_check_tracking_information",
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, custom_llm_provider, expected_document_type, expected_uri_prefix",
    [
        ("mistral/mistral-ocr-latest", "mistral", "document_url", "data:application/pdf;base64,"),
        ("azure_ai/mistral-document-ai-2512", "azure_ai", "document_url", "data:application/pdf;base64,"),
        ("cohere/parse-v5.0", "cohere", "image_url", "data:image/png;base64,"),
        ("azure_ai/Cohere-parse-v5", "azure_ai", "image_url", "data:image/png;base64,"),
    ],
)
async def test_ocr_health_check_sends_the_document_kind_the_provider_config_accepts(
    model, custom_llm_provider, expected_document_type, expected_uri_prefix
):
    handlers = HealthCheckHelpers.get_mode_handlers(
        model=model,
        custom_llm_provider=custom_llm_provider,
        model_params={"model": model, "api_key": "sk-test"},
    )

    with patch(  # test-quality-ok: the public health-check path has no dependency injection seam
        "litellm.aocr", new_callable=AsyncMock, return_value={}
    ) as mock_aocr:
        await handlers["ocr"]()

    document = mock_aocr.call_args.kwargs["document"]
    assert document["type"] == expected_document_type
    assert document[expected_document_type].startswith(expected_uri_prefix)


def test_realtime_health_check_azure_ad_params_drop_reserved_keys():
    from litellm.realtime_api import main as realtime_main

    seen = []
    with patch.object(realtime_main, "get_azure_ad_token", lambda params: seen.append(params) or "ad-token"):
        headers = realtime_main._realtime_health_check_auth_headers(
            "azure",
            None,
            MappingProxyType({"api_base": "https://x.openai.azure.com", "self": 1, "params": 2, "__class__": 3}),
        )

    assert dict(headers) == {"Authorization": "Bearer ad-token"}
    assert seen[0].api_base == "https://x.openai.azure.com"
    assert seen[0].model_extra == {}


def test_ocr_health_check_document_uses_the_native_binding():
    from litellm.litellm_core_utils.health_check_helpers import (
        _ocr_health_check_document,  # pyright: ignore[reportPrivateUsage]  # tests the health-check wiring
    )
    from litellm.rust_bridge.ocr.entrypoints import NATIVE_OCR_HEALTH_CHECK_DOCUMENT

    document: Final = {
        "type": "image_url",
        "image_url": "data:image/png;base64,iVBORw0KGgo=",
    }
    NATIVE_OCR_HEALTH_CHECK_DOCUMENT.override(lambda model, provider: document)
    try:
        assert _ocr_health_check_document(model="mistral/mistral-ocr-latest", custom_llm_provider="mistral") is document
    finally:
        NATIVE_OCR_HEALTH_CHECK_DOCUMENT.reset()


def test_ocr_health_check_document_raises_without_the_extension():
    from litellm.litellm_core_utils.health_check_helpers import (
        _ocr_health_check_document,  # pyright: ignore[reportPrivateUsage]  # tests the health-check wiring
    )
    from litellm.rust_bridge import runtime
    from litellm.rust_bridge.ocr.entrypoints import NATIVE_OCR_HEALTH_CHECK_DOCUMENT

    NATIVE_OCR_HEALTH_CHECK_DOCUMENT.override(None)
    try:
        with pytest.raises(runtime.NoPythonImplementationError):
            _ocr_health_check_document(model="mistral/mistral-ocr-latest", custom_llm_provider="mistral")
    finally:
        NATIVE_OCR_HEALTH_CHECK_DOCUMENT.reset()


@pytest.mark.parametrize(
    ("model", "upstream_url"),
    (
        ("perplexity/pplx-decider-v1-27b", "https://api.perplexity.ai/v1/decisions"),
        ("cloudflare/clef", "https://api.cloudflare.com/client/v4/accounts/acct-1/ai/run/@cf/cloudflare/clef"),
    ),
)
async def test_ahealth_check_probes_evaluation_models_through_the_decisions_api(
    model: str,
    upstream_url: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post(upstream_url).respond(
        json={
            "model": model,
            "answers": {"reachable": {"type": "noul", "noul": 1.0}},
            "usage": {"input_tokens": 12, "output_tokens": 1},
        }
    )

    result: Final = await ahealth_check({"model": model, "api_key": "sk-test"}, mode=None)

    assert "error" not in result, result
    assert upstream.called
    sent: Final = json.loads(upstream.calls[0].request.content)
    assert sent["state"] == "health check"
    assert sent["questions"]["reachable"]["type"] == "noul"


@pytest.mark.asyncio
async def test_ahealth_check_evaluation_uses_configured_probe_state_and_questions(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        json={
            "model": "perplexity/pplx-decider-v1-27b",
            "answers": {"ok": {"type": "noul", "noul": 1.0}},
            "usage": {"input_tokens": 12, "output_tokens": 1},
        }
    )

    result: Final = await ahealth_check(
        {
            "model": "perplexity/pplx-decider-v1-27b",
            "api_key": "sk-test",
            "state": "custom probe",
            "questions": {"ok": {"type": "noul", "instructions": "Is it ok?"}},
        },
        mode=None,
    )

    assert "error" not in result, result
    assert upstream.called
    sent: Final = json.loads(upstream.calls[0].request.content)
    assert sent["state"] == "custom probe"
    assert set(sent["questions"]) == {"ok"}


@pytest.mark.asyncio
async def test_ahealth_check_probes_strands_through_decisions_without_mode(
    local_model_cost_map: None,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("http://strands.local:8080/v1/systemone").respond(
        json={
            "model": "strands-decider-2B-hobson-v19",
            "answers": {"reachable": {"type": "noul", "noul": 1.0}},
            "usage": {"input_tokens": 12, "output_tokens": 1},
        }
    )

    result: Final = await ahealth_check(
        {
            "model": "strands_decider/strands-decider-2B-hobson-v19",
            "api_base": "http://strands.local:8080",
        },
        mode=None,
    )

    assert "error" not in result, result
    assert upstream.called
    assert "authorization" not in upstream.calls[0].request.headers


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected"),
    (
        ("anthropic.claude-haiku-4-5", "bedrock_mantle", "anthropic_messages"),
        ("Anthropic.Claude-Opus-5-5", "bedrock_mantle", "anthropic_messages"),
        ("openai.gpt-oss-120b", "bedrock_mantle", None),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock", None),
        ("claude-haiku-4-5", "anthropic", None),
        ("anthropic.claude-haiku-4-5", None, None),
    ),
)
def test_native_health_check_mode_is_messages_only_for_mantle_claude(
    model: str, custom_llm_provider: str | None, expected: str | None
) -> None:
    assert native_health_check_mode(model=model, custom_llm_provider=custom_llm_provider) == expected


def test_default_health_check_mode_prefers_the_native_surface_over_the_cost_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "model_cost", {"anthropic.claude-haiku-4-5": {"mode": "chat"}})

    assert (
        default_health_check_mode(
            requested_model="bedrock_mantle/anthropic.claude-haiku-4-5",
            model="anthropic.claude-haiku-4-5",
            custom_llm_provider="bedrock_mantle",
        )
        == "anthropic_messages"
    )


@pytest.mark.parametrize(
    ("model_cost", "expected"),
    (
        ({"bedrock_mantle/openai.gpt-oss-120b": {"mode": "responses"}}, "responses"),
        ({"openai.gpt-oss-120b": {"mode": "completion"}}, "completion"),
        (
            {
                "bedrock_mantle/openai.gpt-oss-120b": {"mode": "responses"},
                "openai.gpt-oss-120b": {"mode": "completion"},
            },
            "responses",
        ),
        ({}, "chat"),
    ),
)
def test_default_health_check_mode_falls_back_to_cost_map_then_chat(
    model_cost: dict[str, dict[str, str]], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "model_cost", model_cost)

    assert (
        default_health_check_mode(
            requested_model="bedrock_mantle/openai.gpt-oss-120b",
            model="openai.gpt-oss-120b",
            custom_llm_provider="bedrock_mantle",
        )
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode_kwargs", [{}, {"mode": None}], ids=["omitted", "explicit_none"])
async def test_ahealth_check_probes_mantle_claude_through_messages_without_mode(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    mode_kwargs: dict[str, None],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("https://bedrock-mantle.us-east-2.api.aws/anthropic/v1/messages").respond(
        json={
            "id": "msg_health",
            "type": "message",
            "role": "assistant",
            "model": "anthropic.claude-haiku-4-5",
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    )

    result: Final = await ahealth_check(
        {
            "model": "bedrock_mantle/anthropic.claude-haiku-4-5",
            "api_key": "test-bearer",
            "aws_region_name": "us-east-2",
        },
        prompt="test from litellm",
        **mode_kwargs,
    )

    assert "error" not in result, result
    assert upstream.call_count == 1
    sent: Final = json.loads(upstream.calls.last.request.content)
    assert sent["model"] == "anthropic.claude-haiku-4-5"
    assert sent["max_tokens"] == 16
    assert sent["messages"] == [{"role": "user", "content": "test from litellm"}]


@pytest.mark.asyncio
async def test_ahealth_check_anthropic_messages_mode_keeps_caller_supplied_messages(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("https://bedrock-mantle.us-east-2.api.aws/anthropic/v1/messages").respond(
        json={
            "id": "msg_health",
            "type": "message",
            "role": "assistant",
            "model": "anthropic.claude-haiku-4-5",
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    )

    result: Final = await ahealth_check(
        {
            "model": "bedrock_mantle/anthropic.claude-haiku-4-5",
            "api_key": "test-bearer",
            "aws_region_name": "us-east-2",
            "messages": [{"role": "user", "content": "operator probe"}],
            "max_tokens": 4,
        },
        mode="anthropic_messages",
        prompt="test from litellm",
    )

    assert "error" not in result, result
    sent: Final = json.loads(upstream.calls.last.request.content)
    assert sent["max_tokens"] == 4
    assert sent["messages"] == [{"role": "user", "content": "operator probe"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_params", "expected_error"),
    (
        ({"model": "not-a-provider/some-model"}, "LLM Provider NOT provided"),
        ({"api_key": "test-bearer"}, "model not set"),
    ),
    ids=["unknown_provider", "model_missing"],
)
async def test_ahealth_check_without_mode_reports_the_real_failure(
    model_params: dict[str, str], expected_error: str
) -> None:
    result: Final = await ahealth_check(model_params, prompt="test from litellm")

    assert expected_error in result["error"], result["error"]
    assert "Missing `mode`" not in result["error"]
    assert "raw_request_typed_dict" in result
