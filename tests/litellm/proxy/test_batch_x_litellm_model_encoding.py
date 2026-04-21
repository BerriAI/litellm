"""
Unit tests for batch ID encoding when x-litellm-model header is used.

Verifies that create_batch encodes response IDs with model info so that
retrieve_batch can route back to the correct provider/credentials.
"""

from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy.openai_files_endpoints.common_utils import (
    decode_model_from_file_id,
    get_original_file_id,
)
from litellm.types.utils import LiteLLMBatch


def _make_mock_request(headers: dict) -> MagicMock:
    """Create a mock FastAPI Request with the given headers."""
    mock_request = MagicMock()
    mock_request.headers = headers
    mock_request.query_params = {}
    mock_request.url = MagicMock()
    mock_request.url.port = 4000
    mock_request.method = "POST"
    mock_request.url.path = "/v1/batches"
    return mock_request


def _make_batch_response(
    batch_id: str = "batch_abc123",
    input_file_id: str = "file-input456",
    output_file_id: Optional[str] = None,
    error_file_id: Optional[str] = None,
    status: str = "validating",
) -> LiteLLMBatch:
    """Create a mock LiteLLMBatch response from a provider."""
    return LiteLLMBatch(
        id=batch_id,
        object="batch",
        status=status,
        endpoint="/v1/chat/completions",
        input_file_id=input_file_id,
        completion_window="24h",
        created_at=1234567890,
        output_file_id=output_file_id,
        error_file_id=error_file_id,
    )


@pytest.mark.asyncio
async def test_create_batch_with_x_litellm_model_encodes_batch_id():
    """
    When x-litellm-model header is provided, create_batch should encode the
    response batch_id with model info so retrieve_batch can route correctly.
    """
    from litellm.proxy.batches_endpoints.endpoints import create_batch

    model_name = "my-vllm-model"
    raw_batch_id = "batch_abc123"

    mock_response = _make_batch_response(batch_id=raw_batch_id)
    mock_request = _make_mock_request(headers={"x-litellm-model": model_name})
    mock_fastapi_response = MagicMock()
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.parent_otel_span = None
    mock_user_api_key_dict.user_id = "test_user"

    mock_credentials = {
        "api_key": "sk-test",
        "api_base": "http://vllm:8000",
        "custom_llm_provider": "openai",
    }

    with (
        patch(
            "litellm.proxy.batches_endpoints.endpoints._read_request_body",
            new=AsyncMock(
                return_value={
                    "input_file_id": "file-input456",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                }
            ),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.ProxyBaseLLMRequestProcessing"
        ) as mock_processor_cls,
        patch(
            "litellm.proxy.batches_endpoints.endpoints.get_credentials_for_model",
            return_value=mock_credentials,
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.prepare_data_with_credentials",
        ),
        patch(
            "litellm.acreate_batch",
            new=AsyncMock(return_value=mock_response),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.is_known_model",
            return_value=False,
        ),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "1.0.0"),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            MagicMock(
                post_call_success_hook=AsyncMock(return_value=mock_response),
                update_request_status=AsyncMock(),
            ),
        ),
    ):
        # Setup the mock processor to return data and logging obj
        mock_processor = MagicMock()
        mock_processor.common_processing_pre_call_logic = AsyncMock(
            return_value=(
                {
                    "input_file_id": "file-input456",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                MagicMock(),
            )
        )
        mock_processor_cls.return_value = mock_processor

        response = await create_batch(
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            provider=None,
            user_api_key_dict=mock_user_api_key_dict,
        )

    # The batch_id should be encoded with model info
    assert (
        response.id != raw_batch_id
    ), f"Expected batch_id to be encoded, but got raw ID: {response.id}"
    assert response.id.startswith(
        "batch_"
    ), f"Encoded batch_id should keep batch_ prefix, got: {response.id}"

    # Should be decodable back to the original
    decoded_model = decode_model_from_file_id(response.id)
    assert (
        decoded_model == model_name
    ), f"Expected model '{model_name}' from decoded batch_id, got: {decoded_model}"

    original_id = get_original_file_id(response.id)
    assert (
        original_id == raw_batch_id
    ), f"Expected original ID '{raw_batch_id}', got: {original_id}"


@pytest.mark.asyncio
async def test_create_batch_with_x_litellm_model_encodes_output_and_error_file_ids():
    """
    When a completed batch is returned with output_file_id and error_file_id,
    these should also be encoded with model info.
    """
    from litellm.proxy.batches_endpoints.endpoints import create_batch

    model_name = "my-vllm-model"
    raw_output_file = "file-output789"
    raw_error_file = "file-error012"

    mock_response = _make_batch_response(
        batch_id="batch_abc123",
        output_file_id=raw_output_file,
        error_file_id=raw_error_file,
        status="completed",
    )
    mock_request = _make_mock_request(headers={"x-litellm-model": model_name})
    mock_fastapi_response = MagicMock()
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.parent_otel_span = None
    mock_user_api_key_dict.user_id = "test_user"

    mock_credentials = {
        "api_key": "sk-test",
        "api_base": "http://vllm:8000",
        "custom_llm_provider": "openai",
    }

    with (
        patch(
            "litellm.proxy.batches_endpoints.endpoints._read_request_body",
            new=AsyncMock(
                return_value={
                    "input_file_id": "file-input456",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                }
            ),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.ProxyBaseLLMRequestProcessing"
        ) as mock_processor_cls,
        patch(
            "litellm.proxy.batches_endpoints.endpoints.get_credentials_for_model",
            return_value=mock_credentials,
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.prepare_data_with_credentials",
        ),
        patch(
            "litellm.acreate_batch",
            new=AsyncMock(return_value=mock_response),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.is_known_model",
            return_value=False,
        ),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "1.0.0"),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            MagicMock(
                post_call_success_hook=AsyncMock(return_value=mock_response),
                update_request_status=AsyncMock(),
            ),
        ),
    ):
        mock_processor = MagicMock()
        mock_processor.common_processing_pre_call_logic = AsyncMock(
            return_value=(
                {
                    "input_file_id": "file-input456",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                MagicMock(),
            )
        )
        mock_processor_cls.return_value = mock_processor

        response = await create_batch(
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            provider=None,
            user_api_key_dict=mock_user_api_key_dict,
        )

    # output_file_id should be encoded
    assert decode_model_from_file_id(response.output_file_id) == model_name
    assert get_original_file_id(response.output_file_id) == raw_output_file

    # error_file_id should be encoded
    assert decode_model_from_file_id(response.error_file_id) == model_name
    assert get_original_file_id(response.error_file_id) == raw_error_file


@pytest.mark.asyncio
async def test_create_batch_without_x_litellm_model_returns_raw_ids():
    """
    Without x-litellm-model header, create_batch should NOT encode batch IDs
    (falls through to Scenario 3 / custom_llm_provider fallback).
    """
    from litellm.proxy.batches_endpoints.endpoints import create_batch

    raw_batch_id = "batch_abc123"
    mock_response = _make_batch_response(batch_id=raw_batch_id)
    mock_request = _make_mock_request(headers={})
    mock_fastapi_response = MagicMock()
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.parent_otel_span = None
    mock_user_api_key_dict.user_id = "test_user"

    with (
        patch(
            "litellm.proxy.batches_endpoints.endpoints._read_request_body",
            new=AsyncMock(
                return_value={
                    "input_file_id": "file-input456",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                }
            ),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.ProxyBaseLLMRequestProcessing"
        ) as mock_processor_cls,
        patch(
            "litellm.acreate_batch",
            new=AsyncMock(return_value=mock_response),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.is_known_model",
            return_value=False,
        ),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "1.0.0"),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            MagicMock(
                post_call_success_hook=AsyncMock(return_value=mock_response),
                update_request_status=AsyncMock(),
            ),
        ),
    ):
        mock_processor = MagicMock()
        mock_processor.common_processing_pre_call_logic = AsyncMock(
            return_value=(
                {
                    "input_file_id": "file-input456",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                MagicMock(),
            )
        )
        mock_processor_cls.return_value = mock_processor

        response = await create_batch(
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            provider=None,
            user_api_key_dict=mock_user_api_key_dict,
        )

    # Without x-litellm-model, the batch_id should remain raw
    assert response.id == raw_batch_id
    assert decode_model_from_file_id(response.id) is None


class TestBatchIdRoundTripWithRetrieve:
    """
    Tests that batch IDs encoded during create_batch can be decoded
    correctly during retrieve_batch (Scenario 1: model_from_id).
    """

    def test_encoded_batch_id_is_decoded_for_retrieve(self):
        """
        Simulates the full round-trip: create encodes the ID,
        retrieve decodes it to get the model and original batch_id.
        """
        from litellm.proxy.openai_files_endpoints.common_utils import (
            encode_file_id_with_model,
        )

        model_name = "my-vllm-model"
        raw_batch_id = "batch_vllm_12345"

        # What create_batch does:
        encoded_id = encode_file_id_with_model(
            file_id=raw_batch_id, model=model_name, id_type="batch"
        )

        # What retrieve_batch does:
        decoded_model = decode_model_from_file_id(encoded_id)
        original_id = get_original_file_id(encoded_id)

        assert decoded_model == model_name
        assert original_id == raw_batch_id

    def test_vllm_style_batch_id_roundtrip(self):
        """
        VLLM may return batch IDs in various formats.
        Verify round-trip works for common patterns.
        """
        from litellm.proxy.openai_files_endpoints.common_utils import (
            encode_file_id_with_model,
        )

        test_cases = [
            ("batch_abc123", "vllm-llama3"),
            ("batch_67890", "openai/llama-3-8b"),
            ("batch_some-uuid-here", "my-custom-vllm"),
        ]

        for raw_id, model in test_cases:
            encoded = encode_file_id_with_model(
                file_id=raw_id, model=model, id_type="batch"
            )
            assert encoded.startswith("batch_")
            assert decode_model_from_file_id(encoded) == model
            assert get_original_file_id(encoded) == raw_id
