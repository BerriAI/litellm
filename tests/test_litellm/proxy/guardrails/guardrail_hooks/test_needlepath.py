"""
Unit tests for the Needlepath guardrail.

Tests cover:
- apply_guardrail replaces an eligible message with the selected block, query-
  conditioned on the intent of the tool call that produced it
- target selection: tool outputs by default, system/history opt-in, min-chars
  threshold, the query message itself is never rewritten
- the fail-open contract, one test per decline: empty selection
  (records_selected 0, tokens_after 0, blank or missing rendered_context), a
  gate stand-down, HTTP 402/403/429 and other non-2xx, timeouts and transport
  errors, malformed JSON, and an unexpected response schema. Every one asserts
  the messages come back byte-identical.
"""

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.proxy.guardrails.guardrail_hooks.needlepath.needlepath import (
    _MAX_CONCURRENT_SELECTIONS,
    _MAX_TARGETS_PER_REQUEST,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MIN_CHARS_TO_SELECT,
    DEFAULT_OPERATING_POINT,
    NeedlepathGuardrail,
)
from litellm.types.utils import GenericGuardrailAPIInputs

FAKE_API_BASE = "https://needlepath.example.com"
FAKE_API_KEY = "np_live_0123456789abcdefghijklmnopqr"

TOOL_OUTPUT = "Result row: quarterly filing extract. " * 40  # > 500 chars
SELECTED_BLOCK = "Q3 revenue was 4.2 billion."
USER_QUESTION = "What was Q3 revenue?"

AGENT_MESSAGES = [
    {"role": "system", "content": "You are a filings analyst. " * 40},
    {"role": "user", "content": USER_QUESTION},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "edgar_search", "arguments": '{"cik": "0000320193"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": TOOL_OUTPUT},
]

TOOL_INDEX = 3


def _make_guardrail(**kwargs) -> NeedlepathGuardrail:
    defaults = dict(
        api_base=FAKE_API_BASE,
        api_key=FAKE_API_KEY,
        guardrail_name="needlepath-selection",
        default_on=True,
    )
    defaults.update(kwargs)
    return NeedlepathGuardrail(**defaults)


def _select_response(
    rendered_context: object = SELECTED_BLOCK,
    records_selected: object = 1,
    tokens_after: object = 120,
    gate: object = None,
    status: int = 200,
    omit_rendered_context: bool = False,
) -> MagicMock:
    body = {
        "request_id": "req-1",
        "records_selected": records_selected,
        "records_submitted": 1,
        "records_available": 1,
        "tokens_before": 900,
        "tokens_after": tokens_after,
        "tokens_saved": 780,
        "policy_version": "np-2026-07-r2",
        "gate": {"engaged": True, "reason": "engage:needle"} if gate is None else gate,
    }
    if not omit_rendered_context:
        body["rendered_context"] = rendered_context
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = body
    mock.text = ""
    return mock


def _apply_inputs(messages: list) -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(structured_messages=copy.deepcopy(messages))


@pytest.fixture
def guardrail() -> NeedlepathGuardrail:
    return _make_guardrail()


async def _run(guardrail: NeedlepathGuardrail, post_mock, messages=None) -> GenericGuardrailAPIInputs:
    inputs = _apply_inputs(AGENT_MESSAGES if messages is None else messages)
    with patch.object(guardrail.async_handler, "post", post_mock):
        return await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )


def _assert_byte_identical(result: GenericGuardrailAPIInputs, messages=None) -> None:
    """Every message came back exactly as it went in."""
    assert result["structured_messages"] == (AGENT_MESSAGES if messages is None else messages)


# ── init ──────────────────────────────────────────────────────────────


def test_init_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("NEEDLEPATH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        NeedlepathGuardrail(guardrail_name="needlepath-selection")


def test_init_defaults():
    g = _make_guardrail()
    assert g.needlepath_api_base == FAKE_API_BASE
    assert g.select_tool_outputs is True
    assert g.select_history is False
    assert g.select_system is False
    assert g.min_chars_to_select == DEFAULT_MIN_CHARS_TO_SELECT == 500
    assert g.max_context_tokens == DEFAULT_MAX_CONTEXT_TOKENS == 4000
    # Pinned, not inherited from the service default.
    assert g.operating_point == DEFAULT_OPERATING_POINT == "np-2026-07-r2"


def test_init_rejects_non_http_api_base():
    with pytest.raises(ValueError, match="http or https"):
        _make_guardrail(api_base="file:///etc/passwd")


def test_init_rejects_cloud_metadata_api_base():
    with pytest.raises(ValueError, match="cloud-metadata"):
        _make_guardrail(api_base="http://169.254.169.254")


# ── selection core ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_guardrail_replaces_tool_output_with_rendered_context(
    guardrail: NeedlepathGuardrail,
):
    post = AsyncMock(return_value=_select_response())
    result = await _run(guardrail, post)

    _, call_kwargs = post.call_args
    assert call_kwargs["url"] == f"{FAKE_API_BASE}/v1/context/select"
    assert call_kwargs["headers"]["Authorization"] == f"Bearer {FAKE_API_KEY}"
    payload = call_kwargs["json"]
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert record["text"] == TOOL_OUTPUT
    assert record["kind"] == "tool_result"
    # Title from the tool name, source from the tool_call_id.
    assert record["title"] == "edgar_search"
    assert record["source"] == "call_1"
    # The query is the tool call's intent, not the user question.
    assert payload["task"]["prompt"] == 'edgar_search: {"cik": "0000320193"}'
    assert payload["budget"] == {
        "max_context_tokens": 4000,
        "operating_point": "np-2026-07-r2",
    }
    assert payload["render"] is True
    assert payload["render_format"] == "plain"

    out = result["structured_messages"]
    assert out[TOOL_INDEX]["content"] == SELECTED_BLOCK
    # Only that message changed; every other one is byte-identical.
    assert out[0] == AGENT_MESSAGES[0]
    assert out[1] == AGENT_MESSAGES[1]
    assert out[2] == AGENT_MESSAGES[2]
    # The rewritten message keeps its envelope.
    assert out[TOOL_INDEX]["role"] == "tool"
    assert out[TOOL_INDEX]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_tool_output_without_matching_call_uses_user_question(
    guardrail: NeedlepathGuardrail,
):
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_missing", "content": TOOL_OUTPUT},
    ]
    post = AsyncMock(return_value=_select_response())
    await _run(guardrail, post, messages)

    _, call_kwargs = post.call_args
    assert call_kwargs["json"]["task"]["prompt"] == USER_QUESTION


@pytest.mark.asyncio
async def test_system_and_history_not_selected_by_default(
    guardrail: NeedlepathGuardrail,
):
    post = AsyncMock(return_value=_select_response())
    await _run(guardrail, post)
    # Only the tool output was submitted, even though the system message is
    # comfortably over the character threshold.
    assert post.call_count == 1
    assert post.call_args[1]["json"]["records"][0]["text"] == TOOL_OUTPUT


@pytest.mark.asyncio
async def test_opt_in_system_selects_system_message():
    guardrail = _make_guardrail(select_system=True)
    post = AsyncMock(return_value=_select_response())
    await _run(guardrail, post)

    assert post.call_count == 2
    submitted = {call[1]["json"]["records"][0]["kind"] for call in post.call_args_list}
    assert submitted == {"tool_result", "external_data"}


@pytest.mark.asyncio
async def test_short_messages_skipped(guardrail: NeedlepathGuardrail):
    messages = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "tool", "tool_call_id": "call_1", "content": "short"},
    ]
    post = AsyncMock(return_value=_select_response())
    result = await _run(guardrail, post, messages)

    post.assert_not_called()
    assert result["structured_messages"] == messages


@pytest.mark.asyncio
async def test_response_input_type_passthrough(guardrail: NeedlepathGuardrail):
    inputs = _apply_inputs(AGENT_MESSAGES)
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="response")
    assert result is inputs


# ── fail-open contract ────────────────────────────────────────────────
#
# One test per decline. Each asserts the exact inputs object comes back
# (handlers detect a guardrail edit by identity) and that the messages are
# byte-identical to what arrived.


@pytest.mark.asyncio
async def test_fail_open_zero_records_selected(guardrail: NeedlepathGuardrail):
    post = AsyncMock(return_value=_select_response(records_selected=0))
    inputs = _apply_inputs(AGENT_MESSAGES)
    with patch.object(guardrail.async_handler, "post", post):
        result = await guardrail.apply_guardrail(inputs=inputs, request_data={"model": "gpt-4o"}, input_type="request")
    assert result is inputs
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_zero_tokens_after(guardrail: NeedlepathGuardrail):
    post = AsyncMock(return_value=_select_response(tokens_after=0))
    inputs = _apply_inputs(AGENT_MESSAGES)
    with patch.object(guardrail.async_handler, "post", post):
        result = await guardrail.apply_guardrail(inputs=inputs, request_data={"model": "gpt-4o"}, input_type="request")
    assert result is inputs
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_blank_rendered_context(guardrail: NeedlepathGuardrail):
    post = AsyncMock(return_value=_select_response(rendered_context="   "))
    result = await _run(guardrail, post)
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_missing_rendered_context(guardrail: NeedlepathGuardrail):
    post = AsyncMock(return_value=_select_response(omit_rendered_context=True))
    result = await _run(guardrail, post)
    _assert_byte_identical(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "standdown:flat_gap",
        "standdown:high_drift",
        "standdown:insufficient_candidates",
        "standdown:answerability_low",
        "standdown:gate_error:ValueError",
    ],
)
async def test_fail_open_gate_standdown(guardrail: NeedlepathGuardrail, reason: str):
    """A stand-down is the service saying the full content is what to send.

    The reason string is an open enum, so this matches on the ``standdown:``
    prefix rather than on the individual values.
    """
    post = AsyncMock(return_value=_select_response(gate={"engaged": False, "reason": reason}))
    inputs = _apply_inputs(AGENT_MESSAGES)
    with patch.object(guardrail.async_handler, "post", post):
        result = await guardrail.apply_guardrail(inputs=inputs, request_data={"model": "gpt-4o"}, input_type="request")
    assert result is inputs
    _assert_byte_identical(result)
    # The service was called and answered 200; the decline is ours.
    assert post.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 402, 403, 429, 500, 503])
async def test_fail_open_http_error_status(guardrail: NeedlepathGuardrail, status_code: int):
    """The shared handler raises for status, so every non-2xx arrives as an
    HTTPStatusError carrying the upstream body. None of it reaches the client."""
    request = httpx.Request("POST", f"{FAKE_API_BASE}/v1/context/select")
    response = httpx.Response(status_code, request=request, text="upstream detail")
    post = AsyncMock(side_effect=httpx.HTTPStatusError("boom", request=request, response=response))
    result = await _run(guardrail, post)
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_non_2xx_without_raise_for_status(guardrail: NeedlepathGuardrail):
    """A handler configured not to raise still yields a non-2xx response object."""
    post = AsyncMock(return_value=_select_response(status=429))
    result = await _run(guardrail, post)
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_timeout(guardrail: NeedlepathGuardrail):
    post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    result = await _run(guardrail, post)
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_transport_error(guardrail: NeedlepathGuardrail):
    post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    result = await _run(guardrail, post)
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_malformed_json(guardrail: NeedlepathGuardrail):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.side_effect = ValueError("not json")
    mock.text = "<html>gateway error</html>"
    result = await _run(guardrail, AsyncMock(return_value=mock))
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_unexpected_schema(guardrail: NeedlepathGuardrail):
    """A 200 whose body is well-formed JSON but not the documented object."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = ["not", "an", "object"]
    mock.text = ""
    result = await _run(guardrail, AsyncMock(return_value=mock))
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_fail_open_selection_not_smaller(guardrail: NeedlepathGuardrail):
    """A block that is not shorter than the original could only add tokens."""
    post = AsyncMock(return_value=_select_response(rendered_context=TOOL_OUTPUT + " and more"))
    result = await _run(guardrail, post)
    _assert_byte_identical(result)


@pytest.mark.asyncio
async def test_one_message_declining_does_not_block_another():
    """Selection is per message: a decline on one leaves the other applied."""
    guardrail = _make_guardrail(select_system=True)

    def _by_kind(**kwargs):
        if kwargs["json"]["records"][0]["kind"] == "tool_result":
            return _select_response()
        return _select_response(gate={"engaged": False, "reason": "standdown:flat_gap"})

    post = AsyncMock(side_effect=_by_kind)
    result = await _run(guardrail, post)

    out = result["structured_messages"]
    assert out[TOOL_INDEX]["content"] == SELECTED_BLOCK
    # The system message stood down, so it is byte-identical.
    assert out[0] == AGENT_MESSAGES[0]


# ── fan-out bounds ────────────────────────────────────────────────────


def _many_tool_messages(count: int) -> list:
    """A user question plus `count` eligible tool outputs of strictly increasing size."""
    messages = [{"role": "user", "content": USER_QUESTION}]
    for i in range(count):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                # All above the min-chars threshold; index i is the (i+1)-th smallest.
                "content": "a" * (DEFAULT_MIN_CHARS_TO_SELECT + 20 + i),
            }
        )
    return messages


@pytest.mark.asyncio
async def test_target_cap_selects_only_the_largest_messages(guardrail: NeedlepathGuardrail):
    """Past the per-request cap, only the largest messages are selected.

    The smallest overflow messages come back byte-identical and no service
    call is made for them.
    """
    overflow = 4
    messages = _many_tool_messages(_MAX_TARGETS_PER_REQUEST + overflow)

    post = AsyncMock(return_value=_select_response())
    result = await _run(guardrail, post, messages=messages)

    assert post.await_count == _MAX_TARGETS_PER_REQUEST
    out = result["structured_messages"]
    # Tool messages start at index 1 and grow with index: the first `overflow`
    # are the smallest, so they are the ones left untouched.
    for idx in range(1, 1 + overflow):
        assert out[idx] == messages[idx]
    for idx in range(1 + overflow, len(messages)):
        assert out[idx]["content"] == SELECTED_BLOCK


@pytest.mark.asyncio
async def test_concurrent_selections_are_bounded(guardrail: NeedlepathGuardrail):
    """No more than _MAX_CONCURRENT_SELECTIONS service calls are in flight at once."""
    in_flight = 0
    high_water = 0

    async def _tracking_post(*args, **kwargs):
        nonlocal in_flight, high_water
        in_flight += 1
        high_water = max(high_water, in_flight)
        # Yield twice so every scheduled call gets a chance to start before
        # this one finishes; without the semaphore all of them would overlap.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1
        return _select_response()

    messages = _many_tool_messages(_MAX_TARGETS_PER_REQUEST)
    result = await _run(guardrail, _tracking_post, messages=messages)

    assert high_water <= _MAX_CONCURRENT_SELECTIONS
    assert all(m["content"] == SELECTED_BLOCK for m in result["structured_messages"][1:])
