"""
Unit tests for the Headroom guardrail.

Tests cover:
- apply_guardrail compresses messages via /v1/compress and returns them as structured_messages
- x-headroom-bypass: true header causes guardrail to skip compression
- missing or empty messages are passed through unchanged
- response-type input is passed through unchanged
- /v1/compress HTTP error raises HTTPException
- /v1/compress returning malformed JSON raises HTTPException
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.headroom.headroom import HeadroomGuardrail
from litellm.types.utils import GenericGuardrailAPIInputs

FAKE_API_BASE = "https://headroom.example.com"
FAKE_API_KEY = "test-key"

ORIGINAL_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "A" * 5000},
]
COMPRESSED_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "A" * 500},
]


def _make_guardrail(**kwargs) -> HeadroomGuardrail:
    defaults = dict(
        api_base=FAKE_API_BASE,
        api_key=FAKE_API_KEY,
        guardrail_name="headroom",
        default_on=True,
    )
    defaults.update(kwargs)
    return HeadroomGuardrail(**defaults)


def _make_compress_response(messages: list, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {
        "messages": messages,
        "tokens_before": 1000,
        "tokens_after": 100,
        "compression_ratio": 0.1,
        "transforms_applied": ["router:smart_crusher:0.35"],
    }
    mock.text = ""
    return mock


@pytest.fixture
def guardrail() -> HeadroomGuardrail:
    return _make_guardrail()


@pytest.mark.asyncio
async def test_apply_guardrail_compresses_and_returns_structured_messages(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    assert result.get("structured_messages") == COMPRESSED_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_bypass_header_skips_compression(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )
    request_data = {"proxy_server_request": {"headers": {"x-headroom-bypass": "true"}}}

    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )
        mock_post.assert_not_called()

    assert result.get("structured_messages") == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_response_type_passthrough(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["some response text"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
        )
        mock_post.assert_not_called()

    assert result is inputs


@pytest.mark.asyncio
async def test_apply_guardrail_empty_structured_messages_passthrough(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(texts=["hello"])

    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )
        mock_post.assert_not_called()

    assert result is inputs


@pytest.mark.asyncio
async def test_apply_guardrail_http_error_raises():
    guardrail = _make_guardrail()
    mock_response = _make_compress_response([], status=500)
    mock_response.text = "Internal Server Error"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_apply_guardrail_transport_error_raises():
    guardrail = _make_guardrail()

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502
    assert "unreachable" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_apply_guardrail_missing_messages_key_raises():
    guardrail = _make_guardrail()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tokens_before": 100, "tokens_after": 10}
    mock_response.text = "{}"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_apply_guardrail_empty_compressed_messages_raises():
    guardrail = _make_guardrail()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messages": ["not-a-dict", 42, None],
        "tokens_before": 1000,
        "tokens_after": 0,
        "compression_ratio": 0,
    }
    mock_response.text = "{}"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502
    assert "empty message list" in str(exc_info.value.detail)


def test_init_raises_without_api_base():
    with pytest.raises(ValueError, match="API base URL"):
        HeadroomGuardrail(api_base=None)


def test_bypass_header_case_insensitive():
    guardrail = _make_guardrail()

    for header_value in ("true", "True", "TRUE"):
        data = {
            "proxy_server_request": {"headers": {"x-headroom-bypass": header_value}}
        }
        assert guardrail._should_bypass(data) is True

    data = {"proxy_server_request": {"headers": {"x-headroom-bypass": "false"}}}
    assert guardrail._should_bypass(data) is False

    data = {"proxy_server_request": {"headers": {}}}
    assert guardrail._should_bypass(data) is False

    data = {}
    assert guardrail._should_bypass(data) is False


@pytest.mark.asyncio
async def test_apply_guardrail_sends_model_from_config():
    guardrail = _make_guardrail(model="gpt-4o-mini")
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    call_kwargs = mock_post.call_args
    sent_payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert sent_payload.get("model") == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_apply_guardrail_sends_model_from_request_data_when_no_config_model():
    guardrail = _make_guardrail()
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    call_kwargs = mock_post.call_args
    sent_payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert sent_payload.get("model") == "gpt-4o"
