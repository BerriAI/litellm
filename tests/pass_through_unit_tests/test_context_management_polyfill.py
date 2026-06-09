"""Integration tests for context_management polyfill on /v1/messages adapter path."""

import json
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
    CLEARED_TOOL_RESULT_PLACEHOLDER,
)
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Delta,
    Usage,
)

MODEL = "xai/grok-4"


def _make_history(n_pairs: int, result_filler: str = "x" * 50):
    messages = [{"role": "user", "content": "Compare weather across cities."}]
    for i in range(n_pairs):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"toolu_{i:02d}",
                        "name": "get_weather",
                        "input": {"location": f"City{i}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{i:02d}",
                        "content": f"Result {i}: {result_filler}",
                    }
                ],
            }
        )
    return messages


def _mock_completion_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-test",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="ok"),
            )
        ],
        created=0,
        model="grok-4",
        object="chat.completion",
        usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


async def _mock_streaming_chunks():
    yield ModelResponseStream(
        id="chatcmpl-test",
        created=0,
        model="grok-4",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(role="assistant", content="ok"),
            )
        ],
    )
    yield ModelResponseStream(
        id="chatcmpl-test",
        created=0,
        model="grok-4",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


@pytest.mark.asyncio
async def test_polyfill_round_trip_non_streaming():
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _mock_completion_response()

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        response = await litellm.anthropic.messages.acreate(
            model=MODEL,
            messages=_make_history(n_pairs=5),
            max_tokens=128,
            api_key="sk-test",
            context_management={
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "tool_uses", "value": 1},
                        "keep": {"type": "tool_uses", "value": 2},
                    }
                ]
            },
        )

    # 1. Downstream got the edited messages — older tool_result.content cleared.
    downstream_messages = captured.get("messages")
    assert downstream_messages is not None
    cleared_ids = {"toolu_00", "toolu_01", "toolu_02"}
    kept_ids = {"toolu_03", "toolu_04"}
    found_cleared = 0
    for msg in downstream_messages:
        # The adapter may have translated the messages out of Anthropic shape;
        # we accept either Anthropic-shape (tool_result block) or OpenAI-shape
        # (tool-role message whose content is the placeholder).
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if msg.get("tool_call_id") in cleared_ids:
                content = msg.get("content")
                if isinstance(content, str):
                    if CLEARED_TOOL_RESULT_PLACEHOLDER in content:
                        found_cleared += 1
                elif isinstance(content, list):
                    text = "".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    )
                    if CLEARED_TOOL_RESULT_PLACEHOLDER in text:
                        found_cleared += 1
            elif msg.get("tool_call_id") in kept_ids:
                content = msg.get("content")
                if isinstance(content, str):
                    assert CLEARED_TOOL_RESULT_PLACEHOLDER not in content
    assert found_cleared == 3

    # 2. context_management must not leak into downstream kwargs.
    assert "context_management" not in captured

    # 3. Response carries the applied_edits in Anthropic's documented shape.
    assert isinstance(response, dict)
    cm = response.get("context_management")
    assert cm is not None, f"context_management missing from response: {response}"
    edits = cm.get("applied_edits")
    assert isinstance(edits, list) and len(edits) == 1
    edit = edits[0]
    assert edit["type"] == "clear_tool_uses_20250919"
    assert edit["cleared_tool_uses"] == 3
    assert "cleared_input_tokens" in edit


@pytest.mark.asyncio
async def test_polyfill_trigger_not_met_passes_through_unchanged():
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _mock_completion_response()

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        response = await litellm.anthropic.messages.acreate(
            model=MODEL,
            messages=_make_history(n_pairs=2),
            max_tokens=128,
            api_key="sk-test",
            context_management={
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "input_tokens", "value": 10_000_000},
                        "keep": {"type": "tool_uses", "value": 1},
                    }
                ]
            },
        )

    # Downstream still got the request, but no edits applied.
    assert captured.get("messages") is not None
    assert "context_management" not in captured

    # Response shouldn't carry context_management when nothing fired.
    assert isinstance(response, dict)
    assert (
        response.get("context_management") is None
        or response.get("context_management") == {"applied_edits": []}
        or "context_management" not in response
    )


@pytest.mark.asyncio
async def test_polyfill_streaming_attaches_to_message_delta():
    async def fake_acompletion(**kwargs):
        return _mock_streaming_chunks()

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        response = await litellm.anthropic.messages.acreate(
            model=MODEL,
            messages=_make_history(n_pairs=5),
            max_tokens=128,
            api_key="sk-test",
            stream=True,
            context_management={
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "tool_uses", "value": 1},
                        "keep": {"type": "tool_uses", "value": 2},
                    }
                ]
            },
        )

    # Collect all SSE bytes.
    collected = []
    async for chunk in response:  # type: ignore[union-attr]
        if isinstance(chunk, (bytes, bytearray)):
            collected.append(chunk.decode("utf-8"))
        else:
            collected.append(str(chunk))
    sse_text = "".join(collected)

    # Find the message_delta event payload and check it carries context_management
    # as a sibling of `usage` per Anthropic's spec.
    found_delta_with_cm = False
    for block in sse_text.split("\n\n"):
        if "message_delta" not in block:
            continue
        data_line = next(
            (
                line[len("data:") :].strip()
                for line in block.splitlines()
                if line.startswith("data:")
            ),
            None,
        )
        if data_line is None:
            continue
        payload = json.loads(data_line)
        if payload.get("type") != "message_delta":
            continue
        cm = payload.get("context_management")
        if cm is None:
            continue
        assert "applied_edits" in cm
        assert len(cm["applied_edits"]) == 1
        assert cm["applied_edits"][0]["type"] == "clear_tool_uses_20250919"
        assert cm["applied_edits"][0]["cleared_tool_uses"] == 3
        found_delta_with_cm = True
        break
    assert found_delta_with_cm, (
        "Expected `context_management` on the message_delta SSE event. "
        f"SSE text was: {sse_text!r}"
    )
