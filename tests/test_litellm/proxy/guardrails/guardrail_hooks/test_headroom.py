"""
Unit tests for the Headroom guardrail.

Tests cover:
- apply_guardrail compresses messages via /v1/compress and returns them as structured_messages
- x-headroom-bypass: true header causes guardrail to skip compression
- missing or empty messages are passed through unchanged
- response-type input is passed through unchanged
- /v1/compress HTTP error raises HTTPException
- /v1/compress returning malformed JSON raises HTTPException
- CCR: headroom_retrieve tool injected when compressed messages contain hashes
- CCR: async_should_run_agentic_loop returns True when response has headroom_retrieve tool calls
- CCR: async_build_agentic_loop_plan calls retrieve endpoint and builds follow-up messages
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.headroom.headroom import (
    HeadroomGuardrail,
    extract_hashes_from_messages,
    has_headroom_retrieve_tool,
    HEADROOM_RETRIEVE_TOOL_NAME,
)
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
COMPRESSED_MESSAGES_WITH_HASH = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "Summary. Retrieve more: hash=b573993006976af767214fac",
    },
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


def _make_retrieve_response(original_content: str, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {"original_content": original_content}
    mock.text = original_content
    return mock


def _make_openai_response_with_tool_call(tool_name: str, arguments: dict, tool_id: str = "call_abc123") -> MagicMock:
    fn = MagicMock()
    fn.name = tool_name
    fn.arguments = json.dumps(arguments)

    tc = MagicMock()
    tc.id = tool_id
    tc.type = "function"
    tc.function = fn

    message = MagicMock()
    message.content = None
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


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
async def test_apply_guardrail_injects_retrieve_tool_when_hashes_present(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)

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

    tools = result.get("tools")
    assert tools is not None
    assert has_headroom_retrieve_tool(tools)


@pytest.mark.asyncio
async def test_apply_guardrail_no_tool_injected_when_no_hashes(
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

    tools = result.get("tools")
    assert not has_headroom_retrieve_tool(tools or [])


@pytest.mark.asyncio
async def test_apply_guardrail_preserves_existing_tools_when_injecting(
    guardrail: HeadroomGuardrail,
):
    existing_tool = {"type": "function", "function": {"name": "my_tool", "parameters": {}}}
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
        tools=[existing_tool],
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)

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

    tools = result.get("tools")
    assert tools is not None
    assert isinstance(tools, list)
    assert any(isinstance(t, dict) and t.get("function", {}).get("name") == "my_tool" for t in tools)
    assert has_headroom_retrieve_tool(tools)


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_returns_true_for_retrieve_call(
    guardrail: HeadroomGuardrail,
):
    retrieve_tool_def = [{"type": "function", "function": {"name": HEADROOM_RETRIEVE_TOOL_NAME}}]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "b573993006976af767214fac"},
    )

    should_run, ctx = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=retrieve_tool_def,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is True
    assert len(ctx["tool_calls"]) == 1
    assert ctx["tool_calls"][0]["arguments"]["hash"] == "b573993006976af767214fac"


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_returns_false_without_retrieve_tool(
    guardrail: HeadroomGuardrail,
):
    other_tools = [{"type": "function", "function": {"name": "other_tool"}}]
    response = _make_openai_response_with_tool_call(
        tool_name="other_tool",
        arguments={},
    )

    should_run, _ = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=other_tools,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_returns_false_when_no_retrieve_calls(
    guardrail: HeadroomGuardrail,
):
    retrieve_tool_def = [{"type": "function", "function": {"name": HEADROOM_RETRIEVE_TOOL_NAME}}]
    response = _make_openai_response_with_tool_call(
        tool_name="some_other_function",
        arguments={},
    )

    should_run, _ = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=retrieve_tool_def,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_calls_retrieve_and_builds_messages(
    guardrail: HeadroomGuardrail,
):
    original_content = "This is the full compressed content."
    mock_retrieve = _make_retrieve_response(original_content)

    tool_calls = [
        {
            "id": "call_abc123",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "b573993006976af767214fac"},
        }
    ]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "b573993006976af767214fac"},
        tool_id="call_abc123",
    )
    messages = [{"role": "user", "content": "What does it say? hash=b573993006976af767214fac"}]

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=mock_retrieve,
    ) as mock_get:
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={},
        )

    assert plan.run_agentic_loop is True
    assert plan.request_patch is not None

    follow_up = plan.request_patch.messages
    assert follow_up is not None

    tool_result_message = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result_message is not None
    assert tool_result_message["content"] == original_content
    assert tool_result_message["tool_call_id"] == "call_abc123"

    mock_get.assert_called_once()
    call_url = mock_get.call_args.kwargs.get("url") or mock_get.call_args.args[0]
    assert "b573993006976af767214fac" in call_url


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_handles_retrieve_404(
    guardrail: HeadroomGuardrail,
):
    mock_retrieve = MagicMock()
    mock_retrieve.status_code = 404

    tool_calls = [
        {
            "id": "call_xyz",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "deadbeef000000000000dead"},
        }
    ]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "deadbeef000000000000dead"},
        tool_id="call_xyz",
    )

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=mock_retrieve,
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=[],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={},
        )

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    tool_result = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result is not None
    assert "not found" in tool_result["content"] or "expired" in tool_result["content"]


def test_extract_hashes_from_messages_finds_hashes():
    messages = [
        {"role": "user", "content": "Retrieve more: hash=b573993006976af767214fac"},
        {"role": "assistant", "content": "Also: hash=aabbccdd001122334455aabb"},
    ]
    hashes = extract_hashes_from_messages(messages)
    assert "b573993006976af767214fac" in hashes
    assert "aabbccdd001122334455aabb" in hashes


def test_extract_hashes_from_messages_ignores_short_hashes():
    messages = [{"role": "user", "content": "hash=tooshort"}]
    hashes = extract_hashes_from_messages(messages)
    assert not hashes


def test_extract_hashes_from_list_content_blocks():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hash=b573993006976af767214fac found here"},
            ],
        }
    ]
    hashes = extract_hashes_from_messages(messages)
    assert "b573993006976af767214fac" in hashes


@pytest.mark.asyncio
async def test_apply_guardrail_bypass_header_skips_compression(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )
    request_data = {"proxy_server_request": {"headers": {"x-headroom-bypass": "true"}}}

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
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

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
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

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
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
        data = {"proxy_server_request": {"headers": {"x-headroom-bypass": header_value}}}
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


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_detects_anthropic_content_block_format(
    guardrail: HeadroomGuardrail,
):
    retrieve_tool_def = [{"type": "function", "function": {"name": HEADROOM_RETRIEVE_TOOL_NAME}}]

    response = MagicMock()
    response.choices = None
    response.content = [
        {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "input": {"hash": "b573993006976af767214fac"},
        }
    ]

    should_run, ctx = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="claude-sonnet-4-6",
        messages=[],
        tools=retrieve_tool_def,
        stream=False,
        custom_llm_provider="anthropic",
        kwargs={},
    )

    assert should_run is True
    assert len(ctx["tool_calls"]) == 1
    assert ctx["tool_calls"][0]["arguments"]["hash"] == "b573993006976af767214fac"
