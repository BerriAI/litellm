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
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.compresr.compresr import (
    COMPRESR_RETRIEVE_TOOL_NAME,
    CompresrGuardrail,
    _scoped_store_key,
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


def _make_openai_response_with_tool_calls(tool_calls: list, content: object = None) -> MagicMock:
    """Chat-completion response carrying several tool calls in one turn
    (parallel tool calling). ``tool_calls`` items are (name, arguments, id)."""
    tcs = []
    for name, arguments, tool_id in tool_calls:
        fn = MagicMock()
        fn.name = name
        fn.arguments = json.dumps(arguments)
        tc = MagicMock()
        tc.id = tool_id
        tc.type = "function"
        tc.function = fn
        tcs.append(tc)

    message = MagicMock()
    message.content = content
    message.tool_calls = tcs

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.output = None
    response.content = None
    return response


def _apply_inputs(messages: list) -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(structured_messages=[dict(m) for m in messages])


def _logging_obj(call_id: str) -> SimpleNamespace:
    # Default fixture models a proxy with per-key auth enabled (the production
    # shape). Recovery requires a caller scope; tests that need the no-auth
    # path should build the object explicitly.
    from litellm.proxy._types import UserAPIKeyAuth

    return SimpleNamespace(
        litellm_call_id=call_id,
        model_call_details={
            "litellm_params": {"metadata": {"user_api_key_auth": UserAPIKeyAuth(api_key="hash-default")}}
        },
    )


def _logging_obj_with_key(call_id: str, user_api_key: str, meta_key: str = "metadata") -> SimpleNamespace:
    """Logging object carrying the server-set UserAPIKeyAuth object, the way the
    proxy populates it for an authenticated request (the bare user_api_key
    string alone is never trusted — a client could forge that)."""
    from litellm.proxy._types import UserAPIKeyAuth

    return SimpleNamespace(
        litellm_call_id=call_id,
        model_call_details={"litellm_params": {meta_key: {"user_api_key_auth": UserAPIKeyAuth(api_key=user_api_key)}}},
    )


def _retrieve_tool_call(hash_value: str, tool_id: str) -> dict:
    return {
        "id": tool_id,
        "type": "function",
        "name": COMPRESR_RETRIEVE_TOOL_NAME,
        "arguments": {"hash": hash_value},
    }


def _retrieve_tool_stub() -> dict:
    return {
        "type": "function",
        "function": {"name": COMPRESR_RETRIEVE_TOOL_NAME, "parameters": {}},
    }


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


def test_init_coerces_unknown_unreachable_fallback_to_fail_closed():
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
async def test_apply_guardrail_mirrors_compression_into_texts_channel(
    guardrail: CompresrGuardrail,
):
    """The /v1/responses translation writes compressed output back through the
    `texts` channel, not structured_messages. Compression must be mirrored there
    or that surface silently forwards the original content uncompressed."""
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_unknown", "content": TOOL_OUTPUT},
    ]
    inputs = GenericGuardrailAPIInputs(
        texts=[USER_QUESTION, TOOL_OUTPUT],
        structured_messages=[dict(m) for m in messages],
    )
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )

    texts = result["texts"]
    # The compressed tool output replaces the original in the texts channel...
    assert texts[1].startswith("compressed summary")
    assert texts[1] != TOOL_OUTPUT
    # ...while untouched text passes through byte-identical.
    assert texts[0] == USER_QUESTION


@pytest.mark.asyncio
async def test_apply_guardrail_returns_inputs_unchanged_when_nothing_compressed(
    guardrail: CompresrGuardrail,
):
    """A 200 response whose compressed_context is empty is a functional no-op.
    The exact inputs object must come back: handlers detect guardrail edits by
    identity, and a fresh structured_messages list would force a full write-back
    of an untouched request (on Anthropic, reconversion strips cache_control
    from thinking blocks)."""
    inputs = _apply_inputs(AGENT_MESSAGES)
    mock_post = AsyncMock(return_value=_make_single_compress_response(compressed_context=""))

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )

    assert result is inputs


@pytest.mark.asyncio
async def test_texts_mirror_skips_duplicate_content_with_diverging_compressions(
    guardrail: CompresrGuardrail,
):
    """Two targets with identical text but different query-specific compressions:
    the value-keyed texts mirror cannot tell the occurrences apart, so it must
    leave them uncompressed rather than apply an arbitrary variant to both."""
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "search_docs", "arguments": '{"q": "a"}'}},
                {"id": "call_2", "type": "function", "function": {"name": "search_web", "arguments": '{"q": "b"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": TOOL_OUTPUT},
        {"role": "tool", "tool_call_id": "call_2", "content": TOOL_OUTPUT},
    ]
    inputs = GenericGuardrailAPIInputs(
        texts=[USER_QUESTION, TOOL_OUTPUT, TOOL_OUTPUT],
        structured_messages=[dict(m) for m in messages],
    )
    mock_post = AsyncMock(return_value=_make_batch_compress_response(["compressed for docs", "compressed for web"]))

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )

    # Each message position still gets its own query-specific compression...
    out = result["structured_messages"]
    assert out[2]["content"].startswith("compressed for docs")
    assert out[3]["content"].startswith("compressed for web")
    # ...but the texts mirror leaves the ambiguous occurrences untouched.
    assert result["texts"] == [USER_QUESTION, TOOL_OUTPUT, TOOL_OUTPUT]


@pytest.mark.asyncio
async def test_texts_mirror_skips_text_that_also_appears_outside_targets(
    guardrail: CompresrGuardrail,
):
    """compress_system is off, so a system message whose text happens to equal
    a compressed tool output must not be rewritten through the texts mirror."""
    messages = [
        {"role": "system", "content": TOOL_OUTPUT},
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_x", "content": TOOL_OUTPUT},
    ]
    inputs = GenericGuardrailAPIInputs(
        texts=[TOOL_OUTPUT, USER_QUESTION, TOOL_OUTPUT],
        structured_messages=[dict(m) for m in messages],
    )
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )

    out = result["structured_messages"]
    assert out[0]["content"] == TOOL_OUTPUT  # system message untouched
    assert out[2]["content"].startswith("compressed summary")
    # One compressed target cannot account for two occurrences in texts.
    assert result["texts"] == [TOOL_OUTPUT, USER_QUESTION, TOOL_OUTPUT]


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
async def test_function_result_without_name_does_not_bind_unrelated_call(
    guardrail: CompresrGuardrail,
):
    """A legacy function-role result missing its name must not adopt the intent
    of an arbitrary earlier assistant function_call; it falls back to the last
    user message."""
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        },
        {"role": "function", "content": TOOL_OUTPUT},
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


@pytest.mark.parametrize(
    "api_base",
    [
        "http://2852039166",  # decimal encoding of 169.254.169.254
        "http://0xa9fea9fe",  # hex encoding
        "http://[::ffff:169.254.169.254]",  # IPv4-mapped IPv6
    ],
)
def test_init_rejects_encoded_cloud_metadata_api_base(api_base):
    with pytest.raises(ValueError, match="metadata"):
        CompresrGuardrail(
            api_base=api_base,
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

    assert not any(attacker_call_id in k for k in guardrail._originals_by_call_id)
    assert any(k.endswith("real-framework-call-id") for k in guardrail._originals_by_call_id)


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

    # Attacker's scope resolves nothing, so the loop is vetoed and the victim
    # original never surfaces.
    assert plan.run_agentic_loop is False
    assert plan.request_patch is None


@pytest.mark.asyncio
async def test_recovery_store_partitioned_by_caller_identity(guardrail: CompresrGuardrail):
    """Two tenants that set the SAME client-forgeable x-litellm-call-id must not
    read each other's stored originals, and each still reads its own."""
    shared_call_id = "shared-call-id"
    expected_hash = hashlib.sha256(TOOL_OUTPUT.encode()).hexdigest()[:24]

    async def _plan_for(user_api_key: str, tool_id: str):
        return await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": [_retrieve_tool_call(expected_hash, tool_id)]},
            model="gpt-4o",
            messages=[],
            response=_make_openai_response_with_tool_call(
                COMPRESR_RETRIEVE_TOOL_NAME, {"hash": expected_hash}, tool_id=tool_id
            ),
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=_logging_obj_with_key(shared_call_id, user_api_key),
            stream=False,
            kwargs={},
        )

    # Tenant A compresses and stores its original.
    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=_make_single_compress_response())):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(AGENT_MESSAGES),
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj_with_key(shared_call_id, "hash-tenant-A"),
        )

    # Tenant B, same call id, different virtual-key hash → different bucket, so
    # nothing resolves and the loop is vetoed (Tenant A's original never leaks).
    plan_b = await _plan_for("hash-tenant-B", "call_b")
    assert plan_b.run_agentic_loop is False
    assert plan_b.request_patch is None

    # Tenant A retrieves its own content successfully.
    plan_a = await _plan_for("hash-tenant-A", "call_a")
    assert TOOL_OUTPUT in plan_a.request_patch.messages[-1]["content"]


@pytest.mark.asyncio
async def test_caller_scope_read_from_litellm_metadata(guardrail: CompresrGuardrail):
    """/v1/messages and /v1/responses carry the auth object under
    litellm_metadata rather than metadata; the store key must be scoped by it
    there too, without relying on upstream's metadata backfill."""
    logging_obj = _logging_obj_with_key("call-lm", "hash-tenant-lm", meta_key="litellm_metadata")
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(AGENT_MESSAGES),
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=logging_obj,
        )

    assert "hash-tenant-lm\x00call-lm" in guardrail._originals_by_call_id
    assert "call-lm" not in guardrail._originals_by_call_id


@pytest.mark.asyncio
async def test_caller_scope_rejects_forged_user_api_key_string(guardrail: CompresrGuardrail):
    """A client-supplied metadata.user_api_key STRING (no server-set
    UserAPIKeyAuth object) must not be trusted as a tenant scope — otherwise a
    caller could forge another tenant's recovery bucket on /v1/messages."""
    logging_obj = SimpleNamespace(
        litellm_call_id="call-forge",
        model_call_details={"litellm_params": {"metadata": {"user_api_key": "victim-tenant-hash"}}},
    )
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(AGENT_MESSAGES),
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=logging_obj,
        )

    # Forged string is ignored: scope resolves to empty, so recovery is
    # disabled entirely (no bucket keyed on victim-tenant-hash, no unscoped
    # bucket that another caller could reuse).
    assert not guardrail._originals_by_call_id


@pytest.mark.asyncio
async def test_compress_post_called_with_real_handler_signature():
    """AsyncHTTPHandler.post has a fixed signature; an autospec mock enforces it
    (unlike AsyncMock(spec=...), which silently accepts any kwarg) so a kwarg the
    real handler rejects — which would raise TypeError past the fail policy —
    fails the test instead."""
    guardrail = _make_guardrail()
    autospec_post = create_autospec(guardrail.async_handler.post, return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", autospec_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(AGENT_MESSAGES),
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )

    assert result["structured_messages"][3]["content"].startswith("compressed summary")


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

    tools = result.get("tools")
    assert tools is not None and has_compresr_retrieve_tool(tools)

    scoped_key = next(k for k in guardrail._originals_by_call_id if k.endswith("call-id-1"))
    originals, _expiry = guardrail._originals_by_call_id[scoped_key]
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


@pytest.mark.asyncio
async def test_agentic_plan_returns_stored_original(guardrail: CompresrGuardrail):
    hash_value = hashlib.sha256(TOOL_OUTPUT.encode()).hexdigest()[:24]
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-id-1")), {hash_value: TOOL_OUTPUT})

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
async def test_agentic_plan_preserves_list_shaped_assistant_text(guardrail: CompresrGuardrail):
    """Some providers return chat assistant content as list-of-parts; the
    retrieval follow-up must keep that text, not drop it to None."""
    hash_value = "a" * 24
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-id-1")), {hash_value: "original"})
    response = _make_openai_response_with_tool_calls(
        [(COMPRESR_RETRIEVE_TOOL_NAME, {"hash": hash_value}, "call_r")],
        content=[{"type": "text", "text": "Let me fetch the original."}],
    )

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={"tool_calls": [_retrieve_tool_call(hash_value, "call_r")]},
        model="gpt-4o",
        messages=[{"role": "user", "content": "q"}],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )

    assistant_message = plan.request_patch.messages[-2]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == "Let me fetch the original."


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

    # Hash belongs to another caller's scope; the loop is vetoed and the secret
    # never surfaces.
    assert plan.run_agentic_loop is False
    assert plan.request_patch is None


@pytest.mark.asyncio
async def test_agentic_loop_vetoed_when_no_recovery_state(guardrail: CompresrGuardrail):
    # A caller-defined compresr_retrieve tool with no stored original must not
    # trigger an extra provider round-trip.
    hash_value = "f" * 24
    response = _make_openai_response_with_tool_call(COMPRESR_RETRIEVE_TOOL_NAME, {"hash": hash_value}, tool_id="call_x")
    plan = await guardrail.async_build_agentic_loop_plan(
        tools={"tool_calls": [_retrieve_tool_call(hash_value, "call_x")]},
        model="gpt-4o",
        messages=[{"role": "user", "content": "q"}],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )
    assert plan.run_agentic_loop is False
    assert plan.request_patch is None


@pytest.mark.asyncio
async def test_agentic_loop_dedupes_repeated_retrievals(guardrail: CompresrGuardrail):
    # Retrieving the same marker many times expands the original once; repeats
    # get a short marker (no follow-up amplification).
    hash_value = "a" * 24
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-id-1")), {hash_value: TOOL_OUTPUT})
    calls = [(COMPRESR_RETRIEVE_TOOL_NAME, {"hash": hash_value}, f"call_{i}") for i in range(5)]
    plan = await guardrail.async_build_agentic_loop_plan(
        tools={"tool_calls": [_retrieve_tool_call(hash_value, f"call_{i}") for i in range(5)]},
        model="gpt-4o",
        messages=[{"role": "user", "content": "q"}],
        response=_make_openai_response_with_tool_calls(calls),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )
    tool_results = [m for m in plan.request_patch.messages if m.get("role") == "tool"]
    assert len(tool_results) == 5
    assert sum(1 for m in tool_results if m["content"] == TOOL_OUTPUT) == 1
    assert all("already retrieved" in m["content"] for m in tool_results if m["content"] != TOOL_OUTPUT)


@pytest.mark.asyncio
async def test_agentic_loop_caps_retrieval_count(guardrail: CompresrGuardrail):
    # Beyond _MAX_RETRIEVALS_PER_LOOP retrievals, extra calls get a bounded marker.
    n = 10
    hashes = [f"{i:024x}" for i in range(n)]
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-id-1")), {h: f"original-{h}" for h in hashes})
    calls = [(COMPRESR_RETRIEVE_TOOL_NAME, {"hash": h}, f"call_{i}") for i, h in enumerate(hashes)]
    plan = await guardrail.async_build_agentic_loop_plan(
        tools={"tool_calls": [_retrieve_tool_call(h, f"call_{i}") for i, h in enumerate(hashes)]},
        model="gpt-4o",
        messages=[{"role": "user", "content": "q"}],
        response=_make_openai_response_with_tool_calls(calls),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )
    tool_results = [m for m in plan.request_patch.messages if m.get("role") == "tool"]
    assert len(tool_results) == n
    over_limit = [m for m in tool_results if "retrieval limit reached" in m["content"]]
    assert len(over_limit) == n - 8  # only the first 8 expand


def test_display_hash_strips_control_characters():
    """The compresr_retrieve `hash` argument is model/tool-output-influenced, so
    control characters (newlines, ANSI escapes) must be stripped — not just
    length-capped — before it is echoed into logs or the fallback message."""
    from litellm.proxy.guardrails.guardrail_hooks.compresr.compresr import _display_hash

    assert _display_hash("a" * 24) == "a" * 24  # a real marker hash passes through
    assert "\n" not in _display_hash("abc\ndef\rFORGED LOG LINE")
    assert "\x1b" not in _display_hash("hash\x1b[31mred")
    capped = _display_hash("z" * 100)
    assert capped.endswith("…") and len(capped) <= 33


@pytest.mark.asyncio
async def test_agentic_plan_builds_anthropic_followup_shape(
    guardrail: CompresrGuardrail,
):
    hash_value = "c" * 24
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-id-1")), {hash_value: TOOL_OUTPUT})

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
async def test_agentic_plan_builds_responses_followup_shape(
    guardrail: CompresrGuardrail,
):
    """The /v1/responses path echoes the function_call and pairs it with a
    function_call_output keyed by the same call_id."""
    hash_value = "e" * 24
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-id-1")), {hash_value: TOOL_OUTPUT})

    response = MagicMock()
    response.output = [{"type": "function_call", "call_id": "fc_1"}]  # responses-API shape
    response.content = None

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "fc_1",
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
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )

    call_item, output_item = plan.request_patch.messages[-2], plan.request_patch.messages[-1]
    assert call_item["type"] == "function_call"
    assert call_item["call_id"] == "fc_1"
    assert output_item["type"] == "function_call_output"
    assert output_item["call_id"] == "fc_1"
    assert output_item["output"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_agentic_plan_chat_parallel_tool_calls_echoes_only_retrieve(
    guardrail: CompresrGuardrail,
):
    """When the model calls a real tool alongside compresr_retrieve in one turn,
    only the retrieve call may be echoed in the reconstructed assistant message:
    every echoed tool_call must have a matching tool result or the provider 400s.
    The real call is re-planned by the follow-up; the assistant text is kept."""
    hash_value = "f" * 24
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-id-1")), {hash_value: TOOL_OUTPUT})

    response = _make_openai_response_with_tool_calls(
        [
            ("get_weather", {"city": "Paris"}, "call_weather"),
            (COMPRESR_RETRIEVE_TOOL_NAME, {"hash": hash_value}, "call_retrieve"),
        ],
        content="Let me expand that note and check the weather.",
    )

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={"tool_calls": [_retrieve_tool_call(hash_value, "call_retrieve")]},
        model="gpt-4o",
        messages=[{"role": "user", "content": "q"}],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )

    follow_up = plan.request_patch.messages
    assistant_msg = follow_up[-2]
    echoed_ids = {tc["id"] for tc in assistant_msg["tool_calls"]}
    result_ids = {m["tool_call_id"] for m in follow_up if m.get("role") == "tool"}
    # get_weather is not echoed; every echoed tool_call is answered.
    assert echoed_ids == {"call_retrieve"}
    assert echoed_ids == result_ids
    assert assistant_msg["content"] == "Let me expand that note and check the weather."
    assert follow_up[-1]["content"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_agentic_plan_anthropic_parallel_preserves_text_and_balances(
    guardrail: CompresrGuardrail,
):
    """Anthropic parallel-tool-call turn: the assistant text is preserved, the
    real tool_use is dropped (re-planned), and the reconstructed turn stays
    balanced — one tool_result per echoed tool_use."""
    hash_value = "a" * 23 + "9"
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-id-1")), {hash_value: TOOL_OUTPUT})

    response = MagicMock()
    response.output = None
    response.content = [
        {"type": "text", "text": "Checking the weather and expanding the note."},
        {"type": "tool_use", "id": "toolu_weather", "name": "get_weather", "input": {"city": "Paris"}},
        {
            "type": "tool_use",
            "id": "toolu_retrieve",
            "name": COMPRESR_RETRIEVE_TOOL_NAME,
            "input": {"hash": hash_value},
        },
    ]

    plan = await guardrail.async_build_agentic_loop_plan(
        tools={"tool_calls": [_retrieve_tool_call(hash_value, "toolu_retrieve")]},
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "q"}],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        logging_obj=_logging_obj("call-id-1"),
        stream=False,
        kwargs={},
    )

    assistant_msg, user_msg = plan.request_patch.messages[-2], plan.request_patch.messages[-1]
    assert assistant_msg["content"][0] == {
        "type": "text",
        "text": "Checking the weather and expanding the note.",
    }
    echoed_ids = [b["id"] for b in assistant_msg["content"] if b["type"] == "tool_use"]
    answered_ids = [b["tool_use_id"] for b in user_msg["content"]]
    # get_weather dropped; balanced tool_use/tool_result pairing.
    assert echoed_ids == ["toolu_retrieve"]
    assert answered_ids == echoed_ids


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


def test_originals_store_byte_cap_survives_lone_surrogates():
    # Regression: eviction path must use surrogatepass to match the hash
    # function; a bare encode("utf-8") crashed on lone surrogates.
    guardrail = _make_guardrail(max_bytes_per_call=500)
    surrogate_value = "\ud800" * 60
    hashes = tuple(f"{i:024x}" for i in range(3))
    guardrail._store_originals("c", dict(zip(hashes, (surrogate_value, surrogate_value, surrogate_value))))

    stored, _expiry = guardrail._originals_by_call_id["c"]
    assert hashes[-1] in stored
    assert hashes[0] not in stored


def test_originals_store_caps_total_bytes_across_calls(monkeypatch: pytest.MonkeyPatch):
    # Global byte budget: many distinct call ids must not retain unbounded memory.
    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.compresr.compresr._MAX_TOTAL_STORE_BYTES",
        10_000,
    )
    guardrail = _make_guardrail(max_bytes_per_call=4_000)
    for i in range(20):
        guardrail._store_originals(f"call-{i}", {f"{i:024x}": "x" * 3_000})

    total = sum(
        len(v.encode("utf-8"))
        for originals, _expiry in guardrail._originals_by_call_id.values()
        for v in originals.values()
    )
    assert total <= 10_000
    assert guardrail._store_total_bytes == total  # running counter stays exact
    # Oldest calls evicted; the most-recent call's originals survive.
    assert "call-0" not in guardrail._originals_by_call_id
    assert "call-19" in guardrail._originals_by_call_id


def test_originals_store_global_cap_keeps_current_when_single_call_is_large(
    monkeypatch: pytest.MonkeyPatch,
):
    # One call over the global cap is still kept (only max_bytes_per_call trims it);
    # global eviction never empties the store.
    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.compresr.compresr._MAX_TOTAL_STORE_BYTES",
        1_000,
    )
    guardrail = _make_guardrail(max_bytes_per_call=5_000)
    guardrail._store_originals("solo", {f"{0:024x}": "x" * 4_000})
    assert "solo" in guardrail._originals_by_call_id


# ── dynamic (adaptive) compression — latte_v2 Kneedle ─────────────────


@pytest.mark.asyncio
async def test_dynamic_flag_in_payload():
    """dynamic=True must appear in the compress payload; unset bounds omitted."""
    guardrail = _make_guardrail(dynamic=True)
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_x", "name": "search", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response())
    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["dynamic"] is True
    assert "dynamic_min_ratio" not in payload
    assert "dynamic_max_ratio" not in payload


@pytest.mark.asyncio
async def test_dynamic_bounds_in_payload_when_set():
    guardrail = _make_guardrail(dynamic=True, dynamic_min_ratio=2.0, dynamic_max_ratio=8.0)
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_x", "name": "search", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response())
    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["dynamic"] is True
    assert payload["dynamic_min_ratio"] == 2.0
    assert payload["dynamic_max_ratio"] == 8.0


@pytest.mark.asyncio
async def test_dynamic_on_by_default():
    guardrail = _make_guardrail()  # dynamic defaults on (latte_v2)
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_x", "name": "search", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response())
    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )
    assert mock_post.call_args.kwargs["json"]["dynamic"] is True


# ── generic passthrough compression params ────────────────────────────


@pytest.mark.asyncio
async def test_compression_params_passthrough_in_payload():
    """Extra params in compression_params are forwarded verbatim; named fields
    still win on collision."""
    guardrail = _make_guardrail(compression_params={"heuristic_chunking": True, "coarse": False})
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_x", "name": "search", "content": TOOL_OUTPUT},
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response())
    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["heuristic_chunking"] is True
    # named `coarse` (default True) wins over the passthrough's coarse=False
    assert payload["coarse"] is True


@pytest.mark.asyncio
async def test_compression_params_cannot_override_request_content_fields():
    """context/query/inputs carry the actual content being compressed; a
    passthrough collision on them must be dropped, not silently win."""
    guardrail = _make_guardrail(
        compression_params={
            "context": "injected",
            "query": "injected",
            "inputs": [],
            "heuristic_chunking": True,
        }
    )
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.apply_guardrail(
            inputs=_apply_inputs(AGENT_MESSAGES),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["context"] == TOOL_OUTPUT
    assert payload["query"] == 'web_search: {"query": "2026 EV range"}'
    assert "inputs" not in payload
    assert payload["heuristic_chunking"] is True


# ── compress_last_user ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compress_last_user_compresses_with_verbatim_query():
    """compress_last_user=True compresses the last user message, but the query
    sent to Compresr is still the original verbatim user text."""
    guardrail = _make_guardrail(compress_last_user=True)
    long_question = "Which 2026 EV has the longest range? " * 20  # > 500 chars
    messages = [{"role": "user", "content": long_question}]
    mock_post = AsyncMock(return_value=_make_single_compress_response())

    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["context"] == long_question
    assert payload["query"] == long_question  # verbatim, not the compressed text
    assert result["structured_messages"][0]["content"] == "compressed summary"


# ── malformed-but-200 token stats (must not defeat fail policy) ───────


@pytest.mark.asyncio
async def test_non_numeric_token_stats_do_not_raise(guardrail: CompresrGuardrail):
    """A 200 response with non-numeric token counts must not raise: _call_compress
    already succeeded, so a bare int() here would 500 even under fail policy."""
    resp = _make_single_compress_response()
    resp.json.return_value["data"]["original_tokens"] = "not-a-number"
    resp.json.return_value["data"]["compressed_tokens"] = None

    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=resp)):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(AGENT_MESSAGES),
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    assert result["structured_messages"][3]["content"].startswith("compressed summary")


# ── HTTP status errors (non-2xx from the shared handler) ──────────────


def _http_status_error(status: int = 500, text: str = "upstream error body") -> httpx.HTTPStatusError:
    # The shared AsyncHTTPHandler.post() raises HTTPStatusError on any non-2xx,
    # carrying the upstream body and request headers; this simulates that.
    request = httpx.Request("POST", f"{FAKE_API_BASE}/api/compress/question-specific/")
    response = httpx.Response(status, text=text, request=request)
    return httpx.HTTPStatusError(str(status), request=request, response=response)


@pytest.mark.asyncio
async def test_http_status_error_raises_when_fail_closed(guardrail: CompresrGuardrail):
    with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=_http_status_error(500))):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(AGENT_MESSAGES),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_http_status_error_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    inputs = _apply_inputs(AGENT_MESSAGES)
    with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=_http_status_error(429))):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )
    assert result is inputs
    assert result["structured_messages"][3]["content"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_http_status_error_does_not_leak_upstream_body(guardrail: CompresrGuardrail):
    secret = "SECRET_INSTANCE_METADATA_TOKEN=aws-imds-response"
    with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=_http_status_error(500, text=secret))):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(AGENT_MESSAGES),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )
    assert secret not in json.dumps(exc_info.value.detail)


# ── non-transport httpx errors must still honor the fail policy ────────
# TooManyRedirects and DecodingError are httpx.RequestError but NOT
# httpx.TransportError, so a narrow except would let them escape as a 500
# even under fail_open. These lock in that they are routed through the policy.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.TooManyRedirects("redirect loop"),
        httpx.DecodingError("bad content-encoding"),
    ],
)
async def test_request_errors_raise_when_fail_closed(guardrail: CompresrGuardrail, error):
    with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=error)):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(AGENT_MESSAGES),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.TooManyRedirects("redirect loop"),
        httpx.DecodingError("bad content-encoding"),
    ],
)
async def test_request_errors_fail_open_forwards_uncompressed(error):
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    inputs = _apply_inputs(AGENT_MESSAGES)
    with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=error)):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )
    assert result is inputs
    assert result["structured_messages"][3]["content"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_undecodable_body_on_200_forwards_uncompressed_when_fail_open():
    """A 200 whose body raises DecodingError on .json()/.text must not 500."""
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = httpx.DecodingError("bad content-encoding")
    type(resp).text = property(lambda self: (_ for _ in ()).throw(httpx.DecodingError("bad content-encoding")))
    inputs = _apply_inputs(AGENT_MESSAGES)
    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=resp)):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )
    assert result is inputs
    assert result["structured_messages"][3]["content"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_recursion_error_on_json_forwards_when_fail_open():
    """A deeply nested JSON body can raise RecursionError while parsing; it must
    route through the fail policy, not escape as a 500."""
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = RecursionError("maximum recursion depth exceeded")
    resp.text = ""
    inputs = _apply_inputs(AGENT_MESSAGES)
    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=resp)):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )
    assert result is inputs


@pytest.mark.asyncio
async def test_lone_surrogate_in_content_does_not_crash(guardrail: CompresrGuardrail):
    """A lone Unicode surrogate (reachable via a JSON \\uXXXX escape) in content
    must not crash hashing/byte-accounting after the fail-policy decision."""
    surrogate_output = ("x" * 600) + "\ud800"
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": surrogate_output},
    ]
    mock_post = AsyncMock(return_value=_make_single_compress_response())
    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(messages),
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )
    assert result["structured_messages"][2]["content"].startswith("compressed summary")
    # The original (surrogate included) is recoverable by its hash.
    stored = next(iter(guardrail._originals_by_call_id.values()))[0]
    assert surrogate_output in stored.values()


@pytest.mark.asyncio
async def test_identical_compressed_text_treated_as_noop(guardrail: CompresrGuardrail):
    """If the service returns text byte-identical to the original, nothing
    changed: the exact inputs object is returned so no write-back is forced."""
    inputs = _apply_inputs(AGENT_MESSAGES)
    mock_post = AsyncMock(return_value=_make_single_compress_response(compressed_context=TOOL_OUTPUT))
    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=_logging_obj("call-id-1"),
        )
    assert result is inputs


# ── recovery requires a framework-issued call id ──────────────────────


@pytest.mark.asyncio
async def test_recovery_disabled_without_call_id():
    # enable_retrieval defaults True, but with no framework litellm_call_id we
    # cannot scope stored originals to the request, so compression proceeds
    # without markers, the retrieve tool, or any stored originals.
    guardrail = _make_guardrail()
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


def test_config_model_exposes_unreachable_fallback():
    from litellm.types.proxy.guardrails.guardrail_hooks.compresr import (
        CompresrGuardrailConfigModel,
    )

    field = CompresrGuardrailConfigModel.model_fields.get("unreachable_fallback")
    assert field is not None
    assert field.default == "fail_closed"


# ── audit fixes ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancelled_error_propagates_not_swallowed():
    # Regression: CancelledError is a BaseException, not caught by
    # (RequestError, Timeout). It must re-raise so cooperative cancellation
    # (asyncio.wait_for, client disconnect) still fires.
    import asyncio as _asyncio

    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    with patch.object(guardrail.async_handler, "post", AsyncMock(side_effect=_asyncio.CancelledError())):
        with pytest.raises(_asyncio.CancelledError):
            await guardrail.apply_guardrail(
                inputs=_apply_inputs(AGENT_MESSAGES),
                request_data={"model": "gpt-4o"},
                input_type="request",
            )


def test_max_bytes_per_call_negative_rejected():
    # Regression: a negative value silently disabled the byte cap (< 0 behaves
    # like 0 in _bound_call_bytes). Validate at construction so the footgun
    # surfaces as a ValueError at startup, not silent unbounded storage.
    with pytest.raises(ValueError, match="max_bytes_per_call"):
        _make_guardrail(max_bytes_per_call=-1)


@pytest.mark.asyncio
async def test_max_tokens_zero_from_optional_params_wins_over_kwargs():
    # Regression: `or` short-circuits on falsy values, so an explicit
    # max_tokens=0 from optional_params fell through to kwargs["max_tokens"].
    # Must use `is not None`.
    guardrail = _make_guardrail()
    hash_value = "deadbeef"
    guardrail._store_originals(_scoped_store_key(_logging_obj("call-1")), {hash_value: TOOL_OUTPUT})
    response = MagicMock()
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
        anthropic_messages_optional_request_params={"max_tokens": 0},
        logging_obj=_logging_obj("call-1"),
        stream=False,
        kwargs={"max_tokens": 999},
    )
    assert plan.request_patch.max_tokens == 0


@pytest.mark.asyncio
async def test_recovery_disabled_when_no_caller_scope():
    # Regression: on a no-auth deployment (no UserAPIKeyAuth in metadata) the
    # store key would fall back to the client-settable call id alone, letting
    # any caller retrieve any other caller's originals. Recovery must be off.
    guardrail = _make_guardrail()
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "call-abc"
    logging_obj.model_call_details = {"litellm_params": {"metadata": {}}}
    mock_post = AsyncMock(return_value=_make_single_compress_response())
    with patch.object(guardrail.async_handler, "post", mock_post):
        result = await guardrail.apply_guardrail(
            inputs=_apply_inputs(AGENT_MESSAGES),
            request_data={"model": "gpt-4o"},
            input_type="request",
            logging_obj=logging_obj,
        )
    assert not has_compresr_retrieve_tool(result.get("tools") or [])
    assert guardrail._originals_by_call_id == {}
