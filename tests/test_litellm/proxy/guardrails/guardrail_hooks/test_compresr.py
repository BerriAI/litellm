"""
Unit tests for the Compresr guardrail.

Tests cover:
- apply_guardrail compresses eligible messages query-aware (tool-call intent
  resolved via tool_call_id, falling back to the last user message)
- target selection: tool outputs by default, system/history opt-in, min-chars
  threshold, targets without a derivable query are left uncompressed
- multimodal content: text parts replaced, non-text parts preserved
- recovery: hash marker appended, compresr_retrieve tool injected, originals
  stored per litellm_call_id, agentic loop returns the original content and
  rejects hashes not issued for the current request
- x-compresr-bypass header, response-type passthrough
- fail_closed raises HTTPException; fail_open forwards uncompressed
"""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.compresr.compresr import (
    COMPRESR_RETRIEVE_TOOL_NAME,
    CompresrGuardrail,
    extract_hashes_from_messages,
    has_compresr_retrieve_tool,
)
from litellm.types.utils import GenericGuardrailAPIInputs

FAKE_API_BASE = "https://compresr.example.com"
FAKE_API_KEY = "cmp_test-key"

TOOL_OUTPUT = "Result 1: EV range comparison. " * 40  # > 500 chars
USER_QUESTION = "Which 2026 EV has the longest range?"

AGENT_MESSAGES = [
    {"role": "system", "content": "You are a research assistant."},
    {"role": "user", "content": USER_QUESTION},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "2026 EV range"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": TOOL_OUTPUT},
]


def _make_guardrail(**kwargs) -> CompresrGuardrail:
    defaults = dict(
        api_base=FAKE_API_BASE,
        api_key=FAKE_API_KEY,
        guardrail_name="compresr",
        default_on=True,
    )
    defaults.update(kwargs)
    return CompresrGuardrail(**defaults)


def _make_single_compress_response(
    compressed_context: str = "compressed summary",
    original_tokens: int = 1000,
    compressed_tokens: int = 400,
    status: int = 200,
) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {
        "success": True,
        "data": {
            "compressed_context": compressed_context,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "actual_compression_ratio": 0.6,
            "tokens_saved": original_tokens - compressed_tokens,
            "duration_ms": 42,
        },
    }
    mock.text = ""
    return mock


def _make_batch_compress_response(compressed_contexts: list, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {
        "success": True,
        "data": {
            "results": [
                {
                    "compressed_context": ctx,
                    "original_tokens": 1000,
                    "compressed_tokens": 400,
                    "actual_compression_ratio": 0.6,
                    "tokens_saved": 600,
                    "duration_ms": 42,
                }
                for ctx in compressed_contexts
            ],
            "count": len(compressed_contexts),
        },
    }
    mock.text = ""
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
    # Plain chat-completion shape: no responses-API `output` list, no
    # anthropic `content` list.
    response.output = None
    response.content = None
    return response


def _apply_inputs(messages: list) -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(structured_messages=[dict(m) for m in messages])


def _logging_obj(call_id: str) -> SimpleNamespace:
    return SimpleNamespace(litellm_call_id=call_id)


@pytest.fixture
def guardrail() -> CompresrGuardrail:
    return _make_guardrail()


# ── init ──────────────────────────────────────────────────────────────


def test_init_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        CompresrGuardrail(guardrail_name="compresr")


def test_init_defaults():
    g = _make_guardrail()
    assert g.compresr_api_base == FAKE_API_BASE
    assert g.compression_model == "latte_v2"
    assert g.target_compression_ratio == 0.5
    assert g.coarse is True
    assert g.min_chars_to_compress == 500
    assert g.compress_tool_outputs is True
    assert g.compress_system is False
    assert g.compress_history is False
    assert g.compress_last_user is False
    assert g.enable_retrieval is True
    assert g.unreachable_fallback == "fail_closed"


def test_init_rejects_unknown_unreachable_fallback_value():
    g = _make_guardrail(unreachable_fallback="banana")
    assert g.unreachable_fallback == "fail_closed"


# ── compression core ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_guardrail_compresses_tool_output_with_intent_query(
    guardrail: CompresrGuardrail,
):
    inputs = _apply_inputs(AGENT_MESSAGES)
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )

    _, call_kwargs = mock_post.call_args
    assert call_kwargs["url"] == f"{FAKE_API_BASE}/api/compress/question-specific/"
    assert call_kwargs["headers"]["X-API-Key"] == FAKE_API_KEY
    payload = call_kwargs["json"]
    assert payload["context"] == TOOL_OUTPUT
    # Query is the tool call's intent, not the user question.
    assert payload["query"] == 'web_search: {"query": "2026 EV range"}'
    assert payload["compression_model_name"] == "latte_v2"
    assert payload["target_compression_ratio"] == 0.5

    out = result["structured_messages"]
    assert out[3]["content"].startswith("compressed summary")
    # Untouched messages pass through byte-identical.
    assert out[0] == AGENT_MESSAGES[0]
    assert out[1] == AGENT_MESSAGES[1]
    assert out[2] == AGENT_MESSAGES[2]


@pytest.mark.asyncio
async def test_tool_output_without_matching_call_uses_user_question(
    guardrail: CompresrGuardrail,
):
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_unknown", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["query"] == USER_QUESTION


@pytest.mark.asyncio
async def test_target_without_derivable_query_left_uncompressed(
    guardrail: CompresrGuardrail,
):
    # No user message and no tool-call intent anywhere -> nothing to compress.
    messages = [{"role": "tool", "tool_call_id": "call_x", "content": TOOL_OUTPUT}]
    mock_post = AsyncMock()

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    mock_post.assert_not_called()
    assert result["structured_messages"][0]["content"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_system_and_history_not_compressed_by_default(
    guardrail: CompresrGuardrail,
):
    long_system = "Rules. " * 200
    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": "Old question? " * 100},
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "c1", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    # Only one (single, non-batch) call: the tool output.
    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs["json"]["context"] == TOOL_OUTPUT
    out = result["structured_messages"]
    assert out[0]["content"] == long_system
    assert out[2]["content"] == USER_QUESTION


@pytest.mark.asyncio
async def test_opt_in_system_uses_batch_endpoint():
    guardrail = _make_guardrail(compress_system=True)
    long_system = "Rules. " * 200
    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "c1", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_batch_compress_response(["short system", "short tool"]))

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"].endswith("/api/compress/question-specific/batch")
    batch_inputs = call_kwargs["json"]["inputs"]
    assert [i["context"] for i in batch_inputs] == [long_system, TOOL_OUTPUT]
    out = result["structured_messages"]
    assert out[0]["content"].startswith("short system")
    assert out[2]["content"].startswith("short tool")


@pytest.mark.asyncio
async def test_short_messages_skipped(guardrail: CompresrGuardrail):
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "c1", "content": "tiny result"},
    ]
    mock_post = AsyncMock()

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    mock_post.assert_not_called()
    assert result["structured_messages"][1]["content"] == "tiny result"


@pytest.mark.asyncio
async def test_multimodal_text_replaced_non_text_preserved(
    guardrail: CompresrGuardrail,
):
    image_part = {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"type": "text", "text": TOOL_OUTPUT}, image_part],
        },
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    content = result["structured_messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"].startswith("compressed summary")
    assert content[1] == image_part


# ── passthrough / bypass ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bypass_header_skips_compression_when_allowed():
    guardrail = _make_guardrail(allow_bypass_header=True)
    inputs = _apply_inputs(AGENT_MESSAGES)
    mock_post = AsyncMock()
    request_data = {
        "model": "gpt-4o",
        "proxy_server_request": {"headers": {"x-compresr-bypass": "true"}},
    }

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(inputs=inputs, request_data=request_data, input_type="request")

    mock_post.assert_not_called()
    assert result is inputs


@pytest.mark.asyncio
async def test_bypass_header_ignored_by_default(guardrail: CompresrGuardrail):
    inputs = _apply_inputs(AGENT_MESSAGES)
    mock_post = AsyncMock(return_value=_make_single_compress_response())
    request_data = {
        "model": "gpt-4o",
        "proxy_server_request": {"headers": {"x-compresr-bypass": "true"}},
    }

    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(inputs=inputs, request_data=request_data, input_type="request")

    mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_response_input_type_passthrough(guardrail: CompresrGuardrail):
    inputs = _apply_inputs(AGENT_MESSAGES)
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="response")
    assert result is inputs


@pytest.mark.asyncio
async def test_missing_structured_messages_passthrough(guardrail: CompresrGuardrail):
    inputs = GenericGuardrailAPIInputs(texts=["hello"])
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")
    assert result is inputs


# ── failure policy ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transport_error_raises_when_fail_closed(guardrail: CompresrGuardrail):
    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(side_effect=httpx.ConnectError("boom")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(AGENT_MESSAGES),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_transport_error_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    inputs = _apply_inputs(AGENT_MESSAGES)

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(side_effect=httpx.ConnectError("boom")),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    assert result is inputs
    assert result["structured_messages"][3]["content"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_non_json_response_raises_when_fail_closed(guardrail: CompresrGuardrail):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.side_effect = ValueError("not json")
    mock.text = "<html>gateway error</html>"

    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=mock)):
        with pytest.raises(HTTPException):
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(AGENT_MESSAGES),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )


@pytest.mark.asyncio
async def test_http_exception_does_not_reflect_upstream_body(guardrail: CompresrGuardrail):
    mock = MagicMock()
    mock.status_code = 500
    mock.json.side_effect = ValueError("not json")
    mock.text = "SECRET_INSTANCE_METADATA_TOKEN=aws-imds-response"

    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=mock)):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(AGENT_MESSAGES),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )
    assert "SECRET_INSTANCE_METADATA_TOKEN" not in json.dumps(exc_info.value.detail)


def test_init_rejects_non_http_api_base():
    with pytest.raises(ValueError, match="scheme"):
        CompresrGuardrail(
            api_base="file:///etc/passwd",
            api_key=FAKE_API_KEY,
            guardrail_name="compresr",
        )


def test_init_rejects_cloud_metadata_api_base():
    with pytest.raises(ValueError, match="metadata"):
        CompresrGuardrail(
            api_base="http://169.254.169.254",
            api_key=FAKE_API_KEY,
            guardrail_name="compresr",
        )


def test_ssrf_loopback_blocked_ipv4():
    with pytest.raises(ValueError, match="blocked address"):
        CompresrGuardrail(
            api_base="http://127.0.0.1:8080",
            api_key=FAKE_API_KEY,
            guardrail_name="compresr",
        )


def test_ssrf_loopback_blocked_localhost():
    with pytest.raises(ValueError, match="blocked address"):
        CompresrGuardrail(
            api_base="http://localhost:8080",
            api_key=FAKE_API_KEY,
            guardrail_name="compresr",
        )


def test_ssrf_unspecified_ipv4_blocked():
    with pytest.raises(ValueError, match="blocked address"):
        CompresrGuardrail(
            api_base="http://0.0.0.0:8080",
            api_key=FAKE_API_KEY,
            guardrail_name="compresr",
        )


def test_ssrf_unspecified_ipv6_blocked():
    with pytest.raises(ValueError, match="blocked address"):
        CompresrGuardrail(
            api_base="http://[::]:8080",
            api_key=FAKE_API_KEY,
            guardrail_name="compresr",
        )


@pytest.mark.asyncio
async def test_apply_guardrail_ignores_user_supplied_call_id(guardrail: CompresrGuardrail):
    mock_post = AsyncMock(return_value=_make_single_compress_response())
    attacker_call_id = "victim-tenant-call-id"

    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(AGENT_MESSAGES),
            request_data={"model": "gpt-4o", "litellm_call_id": attacker_call_id},
            input_type="request",
            logging_obj=_logging_obj("real-framework-call-id"),
        )

    assert attacker_call_id not in guardrail._originals_by_call_id
    assert "real-framework-call-id" in guardrail._originals_by_call_id


@pytest.mark.asyncio
async def test_agentic_plan_ignores_user_supplied_call_id(guardrail: CompresrGuardrail):
    hash_value = "d" * 24
    guardrail._store_originals("victim-tenant-call-id", {hash_value: "victim-original"})

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "name": COMPRESR_RETRIEVE_TOOL_NAME,
                    "arguments": {"hash": hash_value},
                }
            ]
        },
        model="gpt-4o",
        messages=[],
        response=_make_openai_response_with_tool_call(
            COMPRESR_RETRIEVE_TOOL_NAME, {"hash": hash_value}, tool_id="call_abc"
        ),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("attacker-call-id"),
        stream=False,
        kwargs={"litellm_call_id": "victim-tenant-call-id"},
    )

    assert "victim-original" not in plan.request_patch.messages[-1]["content"]
    assert "not found" in plan.request_patch.messages[-1]["content"]


@pytest.mark.asyncio
async def test_batch_result_count_mismatch_raises_when_fail_closed():
    guardrail = _make_guardrail(compress_system=True)
    messages = [
        {"role": "system", "content": "Rules. " * 200},
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "c1", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_batch_compress_response(["only one"]))

    with patch.object(guardrail.async_handler, "post", mock_post):
        with pytest.raises(HTTPException):
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(messages),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )


# ── recovery (compresr_retrieve) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_recovery_marker_tool_injection_and_original_stored(
    guardrail: CompresrGuardrail,
):
    inputs = _apply_inputs(AGENT_MESSAGES)
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )

    compressed_content = result["structured_messages"][3]["content"]
    expected_hash = hashlib.sha256(TOOL_OUTPUT.encode()).hexdigest()[:24]
    assert f"compresr hash={expected_hash}" in compressed_content
    assert extract_hashes_from_messages(result["structured_messages"]) == [expected_hash]

    tools = result.get("tools")
    assert tools is not None and has_compresr_retrieve_tool(tools)

    originals, _expiry = guardrail._originals_by_call_id["call-id-1"]
    assert originals[expected_hash] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_enable_retrieval_false_no_marker_no_tool():
    guardrail = _make_guardrail(enable_retrieval=False)
    inputs = _apply_inputs(AGENT_MESSAGES)
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    assert result["structured_messages"][3]["content"] == "compressed summary"
    assert not has_compresr_retrieve_tool(result.get("tools") or [])
    assert guardrail._originals_by_call_id == {}


@pytest.mark.asyncio
async def test_existing_tools_preserved_when_injecting(guardrail: CompresrGuardrail):
    existing_tool = {"type": "function", "function": {"name": "my_tool", "parameters": {}}}
    inputs = GenericGuardrailAPIInputs(
        structured_messages=[dict(m) for m in AGENT_MESSAGES],
        tools=[existing_tool],
    )
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )

    tools = result["tools"]
    assert existing_tool in tools
    assert has_compresr_retrieve_tool(tools)
    assert len(tools) == 2


# ── agentic loop ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_true_for_retrieve_call(
    guardrail: CompresrGuardrail,
):
    response = _make_openai_response_with_tool_call(COMPRESR_RETRIEVE_TOOL_NAME, {"hash": "a" * 24})
    tools = [dict(t) for t in [_retrieve_tool_stub()]]

    should_run, gate_tools = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=tools,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run is True
    assert gate_tools["tool_calls"][0]["name"] == COMPRESR_RETRIEVE_TOOL_NAME


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_false_without_retrieve_tool(
    guardrail: CompresrGuardrail,
):
    response = _make_openai_response_with_tool_call("other_tool", {"x": 1})
    should_run, _ = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=[],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run is False


def _retrieve_tool_stub() -> dict:
    return {
        "type": "function",
        "function": {"name": COMPRESR_RETRIEVE_TOOL_NAME, "parameters": {}},
    }


@pytest.mark.asyncio
async def test_agentic_plan_returns_stored_original(guardrail: CompresrGuardrail):
    hash_value = hashlib.sha256(TOOL_OUTPUT.encode()).hexdigest()[:24]
    guardrail._store_originals("call-id-1", {hash_value: TOOL_OUTPUT})

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "name": COMPRESR_RETRIEVE_TOOL_NAME,
                    "arguments": {"hash": hash_value},
                }
            ]
        },
        model="gpt-4o",
        messages=[{"role": "user", "content": "q"}],
        response=_make_openai_response_with_tool_call(
            COMPRESR_RETRIEVE_TOOL_NAME, {"hash": hash_value}, tool_id="call_abc"
        ),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )

    assert plan.run_agentic_loop is True
    follow_up = plan.request_patch.messages
    tool_result = follow_up[-1]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_abc"
    assert tool_result["content"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_agentic_plan_rejects_hash_from_other_request(
    guardrail: CompresrGuardrail,
):
    hash_value = "b" * 24
    guardrail._store_originals("someone-elses-call", {hash_value: "secret"})

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "name": COMPRESR_RETRIEVE_TOOL_NAME,
                    "arguments": {"hash": hash_value},
                }
            ]
        },
        model="gpt-4o",
        messages=[],
        response=_make_openai_response_with_tool_call(
            COMPRESR_RETRIEVE_TOOL_NAME, {"hash": hash_value}, tool_id="call_abc"
        ),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("my-call"),
        stream=False,
        kwargs={},
    )

    content = plan.request_patch.messages[-1]["content"]
    assert "secret" not in content
    assert "not found" in content


@pytest.mark.asyncio
async def test_agentic_plan_builds_anthropic_followup_shape(
    guardrail: CompresrGuardrail,
):
    hash_value = "c" * 24
    guardrail._store_originals("call-id-1", {hash_value: TOOL_OUTPUT})

    response = MagicMock()
    response.output = None
    response.content = [{"type": "tool_use", "id": "toolu_1"}]

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "name": COMPRESR_RETRIEVE_TOOL_NAME,
                    "arguments": {"hash": hash_value},
                }
            ]
        },
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "q"}],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )

    follow_up = plan.request_patch.messages
    assistant_msg, user_msg = follow_up[-2], follow_up[-1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"][0]["type"] == "tool_use"
    assert user_msg["content"][0]["type"] == "tool_result"
    assert user_msg["content"][0]["tool_use_id"] == "toolu_1"
    assert user_msg["content"][0]["content"] == TOOL_OUTPUT
    assert plan.request_patch.max_tokens == 1024


@pytest.mark.asyncio
async def test_agentic_plan_builds_responses_api_followup_shape(
    guardrail: CompresrGuardrail,
):
    hash_value = "e" * 24
    guardrail._store_originals("call-id-responses", {hash_value: TOOL_OUTPUT})

    # Responses API shape: response.output is a list (not None)
    response = MagicMock()
    response.output = [{"type": "function_call", "call_id": "call_resp_1"}]
    response.content = None

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "call_resp_1",
                    "type": "function",
                    "name": COMPRESR_RETRIEVE_TOOL_NAME,
                    "arguments": {"hash": hash_value},
                }
            ]
        },
        model="gpt-4o",
        messages=[{"role": "user", "content": "q"}],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("call-id-responses"),
        stream=False,
        kwargs={},
    )

    follow_up = plan.request_patch.messages
    # The two injected items are the function_call echo followed by
    # function_call_output carrying the restored content.
    function_call_item = follow_up[-2]
    function_call_output_item = follow_up[-1]
    assert function_call_item["type"] == "function_call"
    assert function_call_item["call_id"] == "call_resp_1"
    assert function_call_output_item["type"] == "function_call_output"
    assert function_call_output_item["call_id"] == "call_resp_1"
    assert function_call_output_item["output"] == TOOL_OUTPUT


# ── store hygiene ─────────────────────────────────────────────────────


def test_originals_store_prunes_expired(guardrail: CompresrGuardrail):
    guardrail._originals_by_call_id["old"] = ({"a" * 24: "x"}, 0.0)  # already expired
    guardrail._store_originals("new", {"b" * 24: "y"})
    assert "old" not in guardrail._originals_by_call_id
    assert "new" in guardrail._originals_by_call_id


def test_originals_store_caps_tracked_calls(guardrail: CompresrGuardrail):
    for i in range(300):
        guardrail._store_originals(f"call-{i}", {("%024x" % i): "x"})
    assert len(guardrail._originals_by_call_id) <= 256
    # Most recent entries survive.
    assert "call-299" in guardrail._originals_by_call_id


def test_originals_store_caps_bytes_per_call():
    guardrail = _make_guardrail(max_bytes_per_call=1000)
    hashes = tuple(f"{i:024x}" for i in range(5))
    values = tuple("x" * 400 for _ in range(5))
    guardrail._store_originals("c", dict(zip(hashes, values)))

    stored, _expiry = guardrail._originals_by_call_id["c"]
    assert sum(len(v.encode("utf-8")) for v in stored.values()) <= 1000
    # Oldest entries are evicted first; newest survives.
    assert hashes[-1] in stored
    assert hashes[0] not in stored


# ── SSRF: DNS rebinding ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ssrf_dns_rebinding_blocked_at_request_time():
    """DNS rebinding: _validate_api_base is called again before each HTTP call."""
    from litellm.proxy.guardrails.guardrail_hooks.compresr import compresr as compresr_mod

    # First call (during __init__) succeeds so the object is created.
    with patch.object(compresr_mod, "_validate_api_base", return_value=FAKE_API_BASE):
        guardrail = _make_guardrail()

    # After creation, DNS flips — subsequent _validate_api_base calls raise.
    with patch.object(
        compresr_mod,
        "_validate_api_base",
        side_effect=ValueError("DNS rebinding detected"),
    ):
        # fail_closed: must raise
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(AGENT_MESSAGES),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )
        assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_ssrf_dns_rebinding_fail_open_forwards_uncompressed():
    from litellm.proxy.guardrails.guardrail_hooks.compresr import compresr as compresr_mod

    with patch.object(compresr_mod, "_validate_api_base", return_value=FAKE_API_BASE):
        guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = _apply_inputs(AGENT_MESSAGES)
    with patch.object(
        compresr_mod,
        "_validate_api_base",
        side_effect=ValueError("DNS rebinding detected"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    # fail_open: original messages must be forwarded uncompressed
    assert result is inputs
    assert result["structured_messages"][3]["content"] == TOOL_OUTPUT


# ── SSRF: Azure IMDS ──────────────────────────────────────────────────


def test_ssrf_azure_imds_blocked():
    with pytest.raises(ValueError, match="metadata"):
        CompresrGuardrail(
            api_base="http://168.63.129.16",
            api_key=FAKE_API_KEY,
            guardrail_name="compresr",
        )


def test_ssrf_metadata_hostname_trailing_dot_blocked():
    """Trailing dot on a blocked hostname must not bypass the blocklist check."""
    with pytest.raises(ValueError, match="metadata"):
        CompresrGuardrail(
            api_base="http://169.254.169.254.",
            api_key=FAKE_API_KEY,
            guardrail_name="compresr",
        )


# ── compression quality guards ────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_guardrail_skips_when_compressed_longer_than_original(
    guardrail: CompresrGuardrail,
):
    """When compressed_context >= original length, keep the original."""
    long_tool_output = "x" * 600
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "c1", "content": long_tool_output},
    ]
    # Return a compressed_context that is longer than the original.
    longer_compressed = "y" * 700
    mock_post = AsyncMock(return_value=_make_single_compress_response(compressed_context=longer_compressed))

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    # Original must be preserved because compression expanded the content.
    assert result["structured_messages"][1]["content"] == long_tool_output


@pytest.mark.asyncio
async def test_apply_guardrail_skips_whitespace_only_compressed_context(
    guardrail: CompresrGuardrail,
):
    """Whitespace-only compressed_context must be treated as empty and skipped."""
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "c1", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response(compressed_context="   \n\t  "))

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    # Whitespace-only result → original must be preserved.
    assert result["structured_messages"][1]["content"] == TOOL_OUTPUT
