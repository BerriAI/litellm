"""
Unit tests for the Headroom guardrail.

Tests cover:
- apply_guardrail compresses messages via /v1/compress and returns them as structured_messages
- x-headroom-bypass: true header causes guardrail to skip compression
- missing or empty messages are passed through unchanged
- response-type input is passed through unchanged
- /v1/compress HTTP error raises HTTPException (fail_closed, the default)
- /v1/compress returning malformed JSON raises HTTPException
- /v1/compress non-2xx surfaces as httpx.HTTPStatusError (raise_for_status),
  not a status_code check on the returned response -- both are handled
- unreachable_fallback="fail_open" forwards the request uncompressed instead of raising
- tokens_saved is derived from tokens_before/tokens_after when the compression
  service omits it, passed through verbatim when present, and skipped (without
  breaking compression) when the token counts are not numeric
- CCR: headroom_retrieve tool injected when compressed messages contain hashes
- CCR: async_should_run_agentic_loop returns True when response has headroom_retrieve tool calls
- CCR: async_build_agentic_loop_plan calls retrieve endpoint and builds follow-up messages
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

import litellm

from litellm.proxy.guardrails.guardrail_hooks.headroom.headroom import (
    HeadroomGuardrail,
    extract_hashes_from_messages,
    has_headroom_retrieve_tool,
    HEADROOM_RETRIEVE_TOOL_NAME,
)
from litellm.proxy.spend_tracking.compression_savings import (
    extract_compression_saved_tokens,
)
from litellm.types.utils import GenericGuardrailAPIInputs

FAKE_API_BASE = "https://headroom.example.com"
FAKE_API_KEY = "test-key"

# The system prompt, the last user turn and the last assistant turn are never
# sent to the compression service, so a fixture needs history for anything to
# be eligible: only index 1 is.
ORIGINAL_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "A" * 5000},
    {"role": "assistant", "content": "Understood."},
    {"role": "user", "content": "and what about B?"},
]
COMPRESSIBLE_MESSAGES = [ORIGINAL_MESSAGES[1]]
COMPRESSED_MESSAGES = [{"role": "user", "content": "A" * 500}]
COMPRESSED_MESSAGES_WITH_HASH = [
    {
        "role": "user",
        "content": "Summary. Retrieve more: hash=b573993006976af767214fac",
    },
]
EXPECTED_MESSAGES = [
    ORIGINAL_MESSAGES[0],
    COMPRESSED_MESSAGES[0],
    ORIGINAL_MESSAGES[2],
    ORIGINAL_MESSAGES[3],
]
EXPECTED_MESSAGES_WITH_HASH = [
    ORIGINAL_MESSAGES[0],
    COMPRESSED_MESSAGES_WITH_HASH[0],
    ORIGINAL_MESSAGES[2],
    ORIGINAL_MESSAGES[3],
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


def _recorded_guardrail_entries(request_data: dict) -> list:
    for container_key in ("metadata", "litellm_metadata"):
        container = request_data.get(container_key)
        if isinstance(container, dict):
            entries = container.get("standard_logging_guardrail_information")
            if isinstance(entries, list):
                return entries
    return []


def _applied_guardrails(request_data: dict) -> list:
    for container_key in ("metadata", "litellm_metadata"):
        container = request_data.get(container_key)
        if isinstance(container, dict) and isinstance(container.get("applied_guardrails"), list):
            return container["applied_guardrails"]
    return []


@pytest.mark.asyncio
async def test_apply_guardrail_compresses_and_returns_structured_messages(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)
    request_data = {"model": "gpt-4o"}

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

    assert result.get("structured_messages") == EXPECTED_MESSAGES

    entries = _recorded_guardrail_entries(request_data)
    assert len(entries) == 1
    assert entries[0]["guardrail_name"] == "headroom"
    assert entries[0]["guardrail_status"] == "success"
    assert entries[0]["guardrail_provider"] == "headroom"
    assert "headroom" in _applied_guardrails(request_data)


def _recorded_guardrail_response(request_data: dict) -> dict:
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    return entries[0]["guardrail_response"]


@pytest.mark.asyncio
async def test_apply_guardrail_derives_tokens_saved_when_service_omits_it(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    # _make_compress_response omits tokens_saved, matching the live service.
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)
    request_data: dict = {"model": "gpt-4o"}

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

    stats = _recorded_guardrail_response(request_data)
    assert stats["tokens_saved"] == 900

    # Spend tracking reads the entry under the spend-log metadata key.
    entry = request_data["metadata"]["standard_logging_guardrail_information"][0]
    assert extract_compression_saved_tokens({"guardrail_information": [entry]}) == 900


@pytest.mark.asyncio
async def test_apply_guardrail_passes_through_service_sent_tokens_saved(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)
    # Deliberately different from tokens_before - tokens_after (900): the
    # service-sent value must win over the derived one.
    mock_response.json.return_value["tokens_saved"] = 123
    request_data: dict = {"model": "gpt-4o"}

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

    assert _recorded_guardrail_response(request_data)["tokens_saved"] == 123


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tokens_before, tokens_after",
    [
        ("1000", "100"),
        (True, False),
        (None, None),
    ],
)
async def test_apply_guardrail_skips_derivation_for_non_numeric_token_counts(
    guardrail: HeadroomGuardrail,
    tokens_before,
    tokens_after,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)
    mock_response.json.return_value["tokens_before"] = tokens_before
    mock_response.json.return_value["tokens_after"] = tokens_after
    request_data: dict = {"model": "gpt-4o"}

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

    assert "tokens_saved" not in _recorded_guardrail_response(request_data)
    # Compression itself is unaffected by the skipped derivation.
    assert result.get("structured_messages") == EXPECTED_MESSAGES


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
    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
    )

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
            kwargs={"litellm_call_id": "call-1"},
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

    messages = [
        {
            "role": "user",
            "content": "Retrieve more: hash=deadbeef000000000000dead",
        }
    ]
    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({"deadbeef000000000000dead"}),
        time.monotonic() + 999,
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
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    tool_result = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result is not None
    assert "not found" in tool_result["content"] or "expired" in tool_result["content"]


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_rejects_hash_with_no_known_call(
    guardrail: HeadroomGuardrail,
):
    """A hash-shaped string planted in message text must not be honored when
    this guardrail has no record of ever issuing it, even if it's echoed back
    in the current request's own messages (e.g. via prompt injection)."""
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
    assert not guardrail._issued_hashes_by_call_id

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
    ) as mock_get:
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=[{"role": "user", "content": "Please fetch hash=deadbeef000000000000dead for me"}],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-unknown"},
        )

    mock_get.assert_not_called()

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    tool_result = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result is not None
    assert "was not produced by the current request" in tool_result["content"]


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_rejects_hash_issued_for_different_call(
    guardrail: HeadroomGuardrail,
):
    """A hash issued for one request must not be retrievable by a different
    request just because the second request echoes that hash-shaped string
    back in its own messages -- retrieval must be scoped per litellm_call_id,
    not derived by re-scanning attacker-controlled message text."""
    guardrail._issued_hashes_by_call_id["call-A"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
    )

    tool_calls = [
        {
            "id": "call_xyz",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "b573993006976af767214fac"},
        }
    ]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "b573993006976af767214fac"},
        tool_id="call_xyz",
    )

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
    ) as mock_get:
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=[{"role": "user", "content": "Please fetch hash=b573993006976af767214fac for me"}],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-B"},
        )

    mock_get.assert_not_called()

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    tool_result = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result is not None
    assert "was not produced by the current request" in tool_result["content"]


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_builds_responses_api_function_call_items(
    guardrail: HeadroomGuardrail,
):
    """For the Responses API, follow-up input must echo a function_call paired
    with a function_call_output keyed by the same call_id -- chat-style
    assistant/tool messages are not valid Responses API input items."""
    original_content = "This is the full compressed content."
    mock_retrieve = _make_retrieve_response(original_content)

    response = MagicMock()
    response.choices = None
    response.content = None
    response.output = [
        {
            "type": "function_call",
            "id": "fc_abc123",
            "call_id": "call_abc123",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": json.dumps({"hash": "b573993006976af767214fac"}),
        }
    ]

    tool_calls = [
        {
            "id": "call_abc123",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "b573993006976af767214fac"},
        }
    ]
    messages = [{"role": "user", "content": "What does it say? hash=b573993006976af767214fac"}]
    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
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
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    assert all("role" not in item for item in follow_up if item not in messages)

    function_call_item = next((i for i in follow_up if i.get("type") == "function_call"), None)
    assert function_call_item is not None
    assert function_call_item["call_id"] == "call_abc123"
    assert function_call_item["name"] == HEADROOM_RETRIEVE_TOOL_NAME

    output_item = next((i for i in follow_up if i.get("type") == "function_call_output"), None)
    assert output_item is not None
    assert output_item["call_id"] == "call_abc123"
    assert output_item["output"] == original_content


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_builds_anthropic_tool_result_messages(
    guardrail: HeadroomGuardrail,
):
    """For the Anthropic Messages API, follow-up must echo a tool_use content
    block in an assistant message paired with a tool_result content block in a
    user message keyed by the same tool_use_id -- chat-style tool-role
    messages are not valid Anthropic input.

    AnthropicMessagesResponse is a TypedDict, so real responses are plain
    dicts at runtime; a MagicMock response here would pass even if branch
    selection used bare getattr() and silently fell through to the
    chat-completions replay shape for every real Anthropic response.
    """
    original_content = "This is the full compressed content."
    mock_retrieve = _make_retrieve_response(original_content)

    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": HEADROOM_RETRIEVE_TOOL_NAME,
                "input": {"hash": "b573993006976af767214fac"},
            }
        ]
    }

    tool_calls = [
        {
            "id": "toolu_abc123",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "b573993006976af767214fac"},
        }
    ]
    messages = [{"role": "user", "content": "What does it say? hash=b573993006976af767214fac"}]
    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
    )

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=mock_retrieve,
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="claude-sonnet-4-5",
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    assert all(m.get("role") != "tool" for m in follow_up)

    assistant_message = next((m for m in follow_up if m.get("role") == "assistant"), None)
    assert assistant_message is not None
    tool_use_block = next((b for b in assistant_message["content"] if b.get("type") == "tool_use"), None)
    assert tool_use_block is not None
    assert tool_use_block["id"] == "toolu_abc123"

    user_message = follow_up[-1]
    assert user_message["role"] == "user"
    tool_result_block = next((b for b in user_message["content"] if b.get("type") == "tool_result"), None)
    assert tool_result_block is not None
    assert tool_result_block["tool_use_id"] == "toolu_abc123"
    assert tool_result_block["content"] == original_content


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


def test_has_headroom_retrieve_tool_recognizes_anthropic_native_shape():
    """By the time an Anthropic Messages API response reaches the agentic-loop
    gate, the OpenAI-shaped tool this guardrail injects (type: "function")
    has already been transformed into Anthropic's native tool shape
    (type: "custom", top-level "name", no nested "function" object)."""
    anthropic_native_tools = [
        {
            "type": "custom",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "input_schema": {"type": "object", "properties": {"hash": {"type": "string"}}},
        }
    ]
    assert has_headroom_retrieve_tool(anthropic_native_tools)
    assert not has_headroom_retrieve_tool([{"type": "custom", "name": "some_other_tool"}])


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
    assert _recorded_guardrail_entries(request_data) == []


@pytest.mark.asyncio
async def test_apply_guardrail_response_type_passthrough(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["some response text"],
        structured_messages=ORIGINAL_MESSAGES,
    )
    request_data: dict = {"model": "gpt-4o"}

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
        )
        mock_post.assert_not_called()

    assert result is inputs
    assert _recorded_guardrail_entries(request_data) == []


@pytest.mark.asyncio
async def test_apply_guardrail_empty_structured_messages_passthrough(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(texts=["hello"])
    request_data: dict = {"model": "gpt-4o"}

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )
        mock_post.assert_not_called()

    assert result is inputs
    assert _recorded_guardrail_entries(request_data) == []
    assert "headroom" not in _applied_guardrails(request_data)


@pytest.mark.asyncio
async def test_passthrough_handler_does_not_log_headroom_as_run(
    guardrail: HeadroomGuardrail,
):
    """Regression for LIT-4650.

    A passthrough request drives headroom through PassThroughEndpointHandler, which
    only supplies `texts` (no `structured_messages`). Headroom cannot compress that
    shape and no-ops, so it must not appear in the spend log's
    standard_logging_guardrail_information as a successful run.
    """
    from litellm.llms.pass_through.guardrail_translation.handler import (
        PassThroughEndpointHandler,
    )

    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        await PassThroughEndpointHandler().process_input_messages(
            data=data,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )
        mock_post.assert_not_called()

    assert _recorded_guardrail_entries(data) == []
    assert "headroom" not in _applied_guardrails(data)


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


def _make_http_status_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", f"{FAKE_API_BASE}/v1/compress")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(
        f"Server error '{status}' for url",
        request=request,
        response=response,
    )


@pytest.mark.asyncio
async def test_apply_guardrail_http_status_error_raises():
    """Regression test: litellm's async httpx client calls raise_for_status()
    internally, so a non-2xx /v1/compress response surfaces as
    httpx.HTTPStatusError, not as a returned MagicMock with status_code set.
    A prior version of _call_compress only checked response.status_code and
    never caught this exception, so it went unhandled instead of blocking
    the request per fail_closed policy."""
    guardrail = _make_guardrail()

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=_make_http_status_error(500, "headroom internal error"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_apply_guardrail_404_error_includes_troubleshooting_hint():
    """404 responses include a troubleshooting hint for self-hosted Headroom deployments."""
    guardrail = _make_guardrail()

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=_make_http_status_error(404, "Not Found"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail["status_code"] == 404
    assert exc_info.value.detail["body"] == "Not Found"
    assert "hint" in exc_info.value.detail
    assert "HEADROOM_COMPRESS_ALLOW_REMOTE=1" in exc_info.value.detail["hint"]


@pytest.mark.asyncio
async def test_apply_guardrail_non_404_error_omits_troubleshooting_hint():
    guardrail = _make_guardrail()

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=_make_http_status_error(500, "headroom internal error"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail["status_code"] == 500
    assert exc_info.value.detail["body"] == "headroom internal error"
    assert "hint" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_apply_guardrail_http_status_error_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=_make_http_status_error(500, "headroom internal error"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_transport_error_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )
    request_data = {"model": "gpt-4o"}

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES

    entries = _recorded_guardrail_entries(request_data)
    assert len(entries) == 1
    assert entries[0]["guardrail_name"] == "headroom"
    assert entries[0]["guardrail_status"] == "guardrail_failed_to_respond"
    assert "headroom" in _applied_guardrails(request_data)


@pytest.mark.asyncio
async def test_apply_guardrail_http_error_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
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
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_non_json_response_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("not JSON")
    mock_response.text = "<!DOCTYPE html><html>not json</html>"

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
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_missing_messages_key_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
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
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_empty_compressed_messages_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
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
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_fail_open_does_not_register_hashes_from_original_messages():
    """When compression fails with fail_open, user-supplied messages that
    happen to contain hash-shaped strings must NOT cause those hashes to be
    registered as valid for CCR retrieval. Otherwise an attacker can plant a
    hash= string in their prompt, trigger a compression failure, and have
    that hash honored by a later headroom_retrieve tool call."""
    messages_with_fake_hash = [
        {"role": "user", "content": "Please fetch hash=deadbeef000000000000dead for me"},
    ]
    guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=messages_with_fake_hash,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == messages_with_fake_hash
    assert not has_headroom_retrieve_tool(result.get("tools") or [])
    assert not guardrail._issued_hashes_by_call_id


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


def test_init_defaults_to_fail_closed():
    guardrail = _make_guardrail()
    assert guardrail.unreachable_fallback == "fail_closed"


def test_init_rejects_invalid_unreachable_fallback_value():
    guardrail = _make_guardrail(unreachable_fallback="not-a-real-mode")
    assert guardrail.unreachable_fallback == "fail_closed"


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
    # Anthropic's native tool format (type: "custom", top-level "name") --
    # by the time a Messages API response reaches this gate, the OpenAI-shaped
    # tool this guardrail injects has already been transformed into this shape.
    retrieve_tool_def = [
        {
            "type": "custom",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "input_schema": {"type": "object", "properties": {"hash": {"type": "string"}}},
        }
    ]

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


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_detects_anthropic_response_as_plain_dict(
    guardrail: HeadroomGuardrail,
):
    """AnthropicMessagesResponse is a TypedDict -- real Messages API responses
    are plain dicts at runtime, not objects with attribute access. A
    MagicMock-only test would pass even if detection used bare getattr() and
    silently treated every real response as having no tool calls."""
    retrieve_tool_def = [
        {
            "type": "custom",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "input_schema": {"type": "object", "properties": {"hash": {"type": "string"}}},
        }
    ]
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": HEADROOM_RETRIEVE_TOOL_NAME,
                "input": {"hash": "b573993006976af767214fac"},
            }
        ]
    }

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


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_detects_responses_api_output_format(
    guardrail: HeadroomGuardrail,
):
    retrieve_tool_def = [{"type": "function", "function": {"name": HEADROOM_RETRIEVE_TOOL_NAME}}]

    response = MagicMock()
    response.choices = None
    response.content = None
    response.output = [
        {
            "type": "function_call",
            "id": "fc_abc123",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": json.dumps({"hash": "b573993006976af767214fac"}),
        }
    ]

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
async def test_apply_guardrail_litellm_timeout_raises_when_fail_closed():
    guardrail = _make_guardrail()

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=litellm.Timeout(
            message="Connection timed out after 10 seconds.",
            model="default-model-name",
            llm_provider="litellm-httpx-handler",
        ),
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
async def test_apply_guardrail_litellm_timeout_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=litellm.Timeout(
            message="Connection timed out after 10 seconds.",
            model="default-model-name",
            llm_provider="litellm-httpx-handler",
        ),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES




# ---------------------------------------------------------------------------
# Content-parts flattening (LIT-4795)
#
# Anthropic-format requests translate to messages whose content is a list of
# part dicts. The compression service only rewrites string content, so the
# guardrail flattens ALL-TEXT part lists on the wire and restores the
# original shapes afterwards. Rows with non-text parts are never flattened:
# cache_control breakpoints are positional, and merging text across a
# non-text part would move a later breakpoint to the other side of it.
# ---------------------------------------------------------------------------

PARTS_MESSAGES = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Earlier turn.", "cache_control": {"type": "ephemeral"}},
            {
                "type": "text",
                "text": "Second block. " + "B" * 5000,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Mixed row text."},
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
        ],
    },
    {"role": "tool", "content": "tool output " + "C" * 500},
    {"role": "user", "content": "what does that file do?"},
]

FLATTENED_HISTORY_TEXT = "Earlier turn.\n\nSecond block. " + "B" * 5000


def _parts_copy() -> list:
    return json.loads(json.dumps(PARTS_MESSAGES))


def _echo_wire_view() -> list:
    """What the service receives (and echoes back when it changes nothing).

    The system row and the trailing user row are never sent.
    """
    return [
        {"role": "user", "content": FLATTENED_HISTORY_TEXT},
        json.loads(json.dumps(PARTS_MESSAGES[2])),
        {"role": "tool", "content": "tool output " + "C" * 500},
    ]


@pytest.mark.asyncio
async def test_apply_guardrail_flattens_all_text_rows_only(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["B" * 5000],
        structured_messages=_parts_copy(),
    )
    mock_response = _make_compress_response(_echo_wire_view())

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "claude-fable-5"},
            input_type="request",
        )

    wire_messages = mock_post.call_args.kwargs["json"]["messages"]
    assert wire_messages[0]["content"] == FLATTENED_HISTORY_TEXT
    # Mixed text+image row is never flattened: merging its text would move a
    # later cache_control breakpoint across the image part.
    assert isinstance(wire_messages[1]["content"], list)
    assert wire_messages[2]["content"] == "tool output " + "C" * 500


@pytest.mark.asyncio
async def test_apply_guardrail_restores_rewritten_all_text_row(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["B" * 5000],
        structured_messages=_parts_copy(),
    )
    compressed = _echo_wire_view()
    compressed[0]["content"] = "compressed history. Retrieve more: hash=b573993006976af767214fac"
    mock_response = _make_compress_response(compressed)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "claude-fable-5"},
            input_type="request",
        )

    messages = result["structured_messages"]
    history_content = messages[1]["content"]
    # Rewritten all-text row collapses to one part carrying the LAST declared
    # breakpoint: an Anthropic breakpoint caches the prefix ending at its
    # part, so after the merge the last one (and its TTL) still describes the
    # row.
    assert isinstance(history_content, list)
    assert len(history_content) == 1
    assert history_content[0]["text"] == "compressed history. Retrieve more: hash=b573993006976af767214fac"
    assert history_content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # Mixed row passes through byte-identical.
    assert messages[2]["content"] == PARTS_MESSAGES[2]["content"]
    # Hashes inside restored parts still drive retrieve-tool injection.
    assert has_headroom_retrieve_tool(result.get("tools") or [])


@pytest.mark.asyncio
async def test_apply_guardrail_keeps_originals_when_service_echoes_unchanged(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["B" * 5000],
        structured_messages=_parts_copy(),
    )
    mock_response = _make_compress_response(_echo_wire_view())

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "claude-fable-5"},
            input_type="request",
        )

    messages = result["structured_messages"]
    assert [m["content"] for m in messages] == [m["content"] for m in PARTS_MESSAGES]


@pytest.mark.asyncio
async def test_apply_guardrail_rejects_service_output_when_rows_dropped(
    guardrail: HeadroomGuardrail,
):
    """A reshaped conversation cannot be applied at all: the rows held back from
    compression are matched positionally, so a response with a different row
    count goes through the fail policy instead of being adopted."""
    inputs = GenericGuardrailAPIInputs(
        texts=["B" * 5000],
        structured_messages=_parts_copy(),
    )
    dropped = [{"role": "user", "content": "B" * 50}]
    mock_response = _make_compress_response(dropped)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"model": "claude-fable-5"},
                input_type="request",
            )

    assert exc_info.value.status_code == 502
    assert "changed the message count" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_apply_guardrail_forwards_original_when_rows_dropped_and_fail_open():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    original = _parts_copy()
    inputs = GenericGuardrailAPIInputs(texts=["B" * 5000], structured_messages=original)
    mock_response = _make_compress_response([{"role": "user", "content": "B" * 50}])

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "claude-fable-5"},
            input_type="request",
        )

    # Same object back, so translation handlers that detect a rewrite by
    # identity leave the request alone instead of round-tripping it.
    assert result is inputs
    assert result["structured_messages"] is original


@pytest.mark.asyncio
async def test_apply_guardrail_sends_textless_parts_rows_unflattened(
    guardrail: HeadroomGuardrail,
):
    image_only = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]},
        {"role": "user", "content": "D" * 5000},
    ]
    inputs = GenericGuardrailAPIInputs(
        texts=["D" * 5000],
        structured_messages=json.loads(json.dumps(image_only)) + [{"role": "user", "content": "and now?"}],
    )
    mock_response = _make_compress_response(json.loads(json.dumps(image_only)))

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "claude-fable-5"},
            input_type="request",
        )

    wire_messages = mock_post.call_args.kwargs["json"]["messages"]
    assert isinstance(wire_messages[0]["content"], list)
    assert wire_messages[1]["content"] == "D" * 5000


@pytest.mark.asyncio
async def test_fail_open_returns_original_parts_shapes():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    inputs = GenericGuardrailAPIInputs(
        texts=["B" * 5000],
        structured_messages=_parts_copy(),
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("boom"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    messages = result["structured_messages"]
    assert [m["content"] for m in messages] == [m["content"] for m in PARTS_MESSAGES]


# ---------------------------------------------------------------------------
# LIT-5018: the turn the model is being asked to act on is never compressed.
#
# A Claude Code request ends with the live instruction, preceded by the tool
# result answering the assistant's last tool call. Replacing either with a
# marker makes the model answer a retrieval result instead of the request.
# ---------------------------------------------------------------------------

AGENTIC_MESSAGES = [
    {"role": "system", "content": "You are Claude Code. " + "S" * 5000},
    {"role": "user", "content": "H" * 5000},
    {"role": "assistant", "content": "Older answer. " + "O" * 5000},
    {"role": "tool", "tool_call_id": "old_1", "content": "older tool output " + "T" * 5000},
    {
        "role": "assistant",
        "content": "Reading the file now.",
        "tool_calls": [{"id": "tu_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "tu_1", "content": "FILE BODY " + "F" * 5000},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "<team expansion> " + "E" * 5000},
            {"type": "text", "text": "can we run /team to fix this"},
        ],
    },
]


async def _wire_and_result(guardrail: HeadroomGuardrail, messages: list, returned: list | None = None):
    inputs = GenericGuardrailAPIInputs(texts=["x"], structured_messages=json.loads(json.dumps(messages)))
    sent: dict = {}

    def _echo(**kwargs):
        sent["messages"] = kwargs["json"]["messages"]
        return _make_compress_response(
            returned if returned is not None else json.loads(json.dumps(kwargs["json"]["messages"]))
        )

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock, side_effect=_echo):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "claude-sonnet-4-5-20250929"},
            input_type="request",
        )
    return sent["messages"], result


@pytest.mark.asyncio
async def test_live_user_turn_is_never_sent_for_compression(guardrail: HeadroomGuardrail):
    wire, result = await _wire_and_result(guardrail, AGENTIC_MESSAGES)

    live_turn = AGENTIC_MESSAGES[-1]
    assert live_turn not in wire
    assert not any("can we run /team to fix this" in json.dumps(row) for row in wire)
    # It reaches the model byte-identical, both text parts intact, so no
    # marker and no retrieval round-trip stands in for the instruction.
    assert result["structured_messages"][-1] == live_turn


@pytest.mark.asyncio
async def test_system_prompt_is_never_sent_for_compression(guardrail: HeadroomGuardrail):
    wire, result = await _wire_and_result(guardrail, AGENTIC_MESSAGES)

    assert not any(row.get("role") == "system" for row in wire)
    # The Anthropic write-back drops compressed system rows, so sending it
    # only inflates the savings the service reports back.
    assert result["structured_messages"][0] == AGENTIC_MESSAGES[0]


@pytest.mark.asyncio
async def test_trailing_tool_exchange_is_never_sent_for_compression(guardrail: HeadroomGuardrail):
    """The tool result answering the last assistant's tool call is protected
    with it: a marker there stands in for the result of the call the model just
    made, forcing an immediate retrieval of data it already asked for."""
    wire, result = await _wire_and_result(guardrail, AGENTIC_MESSAGES)

    assert not any(row.get("tool_call_id") == "tu_1" for row in wire)
    assert result["structured_messages"][5] == AGENTIC_MESSAGES[5]


@pytest.mark.asyncio
async def test_history_is_still_compressed(guardrail: HeadroomGuardrail):
    """Negative control: protection must not turn compression into a no-op."""
    compressed_history = [
        {"role": "user", "content": "hist. hash=b573993006976af767214fac"},
        {"role": "assistant", "content": "older. hash=a73993006976af767214fac1"},
        {"role": "tool", "tool_call_id": "old_1", "content": "older tool. hash=c73993006976af767214fac2"},
    ]
    wire, result = await _wire_and_result(guardrail, AGENTIC_MESSAGES, returned=compressed_history)

    # Exactly the three history rows go to the service, in order.
    assert [row["role"] for row in wire] == ["user", "assistant", "tool"]
    assert wire[0]["content"] == "H" * 5000
    assert wire[2]["tool_call_id"] == "old_1"

    messages = result["structured_messages"]
    assert len(messages) == len(AGENTIC_MESSAGES)
    assert messages[1] == compressed_history[0]
    assert messages[2] == compressed_history[1]
    assert messages[3] == compressed_history[2]
    # Hashes in the compressed history still drive retrieve-tool injection.
    assert has_headroom_retrieve_tool(result.get("tools") or [])


@pytest.mark.asyncio
async def test_nothing_compressible_returns_inputs_untouched(guardrail: HeadroomGuardrail):
    """A single-turn request is all protected, so there is nothing to send and
    the caller's own inputs object comes back."""
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "A" * 5000},
        ],
    )

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    mock_post.assert_not_called()
    assert result is inputs


@pytest.mark.asyncio
async def test_fail_open_returns_the_caller_inputs_object():
    """Translation handlers detect a rewrite by object identity, so a request
    that was not compressed must come back as the same object or it is
    round-tripped through the write-back for nothing."""
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    original = json.loads(json.dumps(AGENTIC_MESSAGES))
    inputs = GenericGuardrailAPIInputs(texts=["x"], structured_messages=original)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("boom"),
    ):
        result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert result is inputs
    assert result["structured_messages"] is original


# ---------------------------------------------------------------------------
# LIT-5018: the retrieval follow-up keeps the model's own text.
# ---------------------------------------------------------------------------


def _anthropic_response_with_text_and_tool_call() -> dict:
    return {
        "content": [
            {"type": "text", "text": "Let me pull the original back."},
            {"type": "tool_use", "id": "call_1", "name": HEADROOM_RETRIEVE_TOOL_NAME, "input": {"hash": "h" * 24}},
        ]
    }


async def _plan_for(guardrail: HeadroomGuardrail, response, messages: list):
    guardrail._issued_hashes_by_call_id["call-1"] = (frozenset({"h" * 24}), time.monotonic() + 60)
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "call-1"
    logging_obj.model_call_details = {}
    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=_make_retrieve_response("ORIGINAL CONTENT"),
    ):
        return await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": [{"id": "call_1", "name": HEADROOM_RETRIEVE_TOOL_NAME, "arguments": {"hash": "h" * 24}}]},
            model="claude-sonnet-4-5-20250929",
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=logging_obj,
            stream=False,
            kwargs={},
        )


@pytest.mark.asyncio
async def test_anthropic_followup_preserves_assistant_text(guardrail: HeadroomGuardrail):
    plan = await _plan_for(guardrail, _anthropic_response_with_text_and_tool_call(), [{"role": "user", "content": "q"}])

    assistant = plan.request_patch.messages[-2]  # type: ignore[union-attr]
    assert assistant["role"] == "assistant"
    # Text first, then the tool_use it accompanied: dropping it loses the
    # model's stated reason for the retrieval from its own transcript.
    assert assistant["content"][0] == {"type": "text", "text": "Let me pull the original back."}
    assert assistant["content"][1]["type"] == "tool_use"


@pytest.mark.asyncio
async def test_responses_followup_preserves_assistant_text(guardrail: HeadroomGuardrail):
    response = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "Fetching the original."}]},
            {"type": "function_call", "call_id": "call_1", "name": HEADROOM_RETRIEVE_TOOL_NAME, "arguments": "{}"},
        ]
    }

    plan = await _plan_for(guardrail, response, [{"role": "user", "content": "q"}])

    items = plan.request_patch.messages  # type: ignore[union-attr]
    assert items[1] == {"role": "assistant", "content": "Fetching the original."}
    assert items[2]["type"] == "function_call"


@pytest.mark.asyncio
async def test_chat_followup_echoes_only_the_retrieve_call(guardrail: HeadroomGuardrail):
    """A turn that called another tool alongside headroom_retrieve must not
    echo that call: only the retrieve call gets a tool result, and a tool_call
    without one is rejected by the provider."""
    other = MagicMock()
    other.id = "call_other"
    other.type = "function"
    other.function = MagicMock()
    other.function.name = "Write"
    other.function.arguments = "{}"

    response = _make_openai_response_with_tool_call(HEADROOM_RETRIEVE_TOOL_NAME, {"hash": "h" * 24}, "call_1")
    response.choices[0].message.content = "Getting the original first."
    response.choices[0].message.tool_calls = [response.choices[0].message.tool_calls[0], other]

    plan = await _plan_for(guardrail, response, [{"role": "user", "content": "q"}])

    messages = plan.request_patch.messages  # type: ignore[union-attr]
    assistant = messages[1]
    assert assistant["content"] == "Getting the original first."
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_1"]
    assert [m["tool_call_id"] for m in messages[2:]] == ["call_1"]
