"""
Tests for advisor orchestration cost aggregation and per-iteration usage
reporting on the Anthropic ``/v1/messages`` path.

Covers:
1. Cost aggregation across executor + advisor sub-calls → final
   ``_hidden_params["response_cost"]`` equals the sum.
2. ``usage.iterations[]`` reports per-call token breakdowns in order.
3. ``litellm_logging_obj.set_cost_breakdown`` is called with the
   "Main Model (initial)" + "Advisor Model" additional costs.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ADVISOR_TOOL = {
    "type": "advisor_20260301",
    "name": "advisor",
    "model": "claude-opus-4-6",
}

MESSAGES = [{"role": "user", "content": "Help me plan this feature."}]


def _make_executor_tool_use_response(
    tool_id: str = "toolu_advisor_01",
    input_tokens: int = 100,
    output_tokens: int = 20,
    response_cost: float = 1.0,
) -> Dict[str, Any]:
    return {
        "id": "msg_executor_toolcall",
        "type": "message",
        "role": "assistant",
        "model": "openai/gpt-4o-mini",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "consult_advisor",
                "input": {"question": "What approach should I use?"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "_hidden_params": {"response_cost": response_cost},
    }


def _make_advisor_text_response(
    text: str = "Use trial division up to sqrt(n).",
    input_tokens: int = 200,
    output_tokens: int = 150,
    response_cost: float = 0.3,
) -> Dict[str, Any]:
    return {
        "id": "msg_advisor",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "_hidden_params": {"response_cost": response_cost},
    }


def _make_final_executor_response(
    text: str = "Here's the implementation.",
    input_tokens: int = 300,
    output_tokens: int = 40,
    response_cost: float = 0.7,
) -> Dict[str, Any]:
    return {
        "id": "msg_executor_final",
        "type": "message",
        "role": "assistant",
        "model": "openai/gpt-4o-mini",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "_hidden_params": {"response_cost": response_cost},
    }


@pytest.mark.asyncio
async def test_advisor_orchestration_aggregates_cost_and_iterations():
    """
    Executor calls advisor once then produces the final response.
    - Total cost = first executor (1.0) + advisor (0.3) + final executor (0.7) = 2.0
    - ``usage.iterations`` contains 3 entries in order.
    - ``set_cost_breakdown`` is called with ``Main Model (initial)`` and
      ``Advisor Model`` entries.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    first_executor_resp = _make_executor_tool_use_response(response_cost=1.0)
    advisor_resp = _make_advisor_text_response(response_cost=0.3)
    final_executor_resp = _make_final_executor_response(response_cost=0.7)

    executor_call_count = 0

    async def mock_messages(model, messages, tools, stream, max_tokens, **kwargs):
        nonlocal executor_call_count
        executor_call_count += 1
        if executor_call_count == 1:
            return first_executor_resp
        return final_executor_resp

    fake_logging_obj = MagicMock()
    fake_logging_obj.model_call_details = {}
    fake_logging_obj.set_cost_breakdown = MagicMock()

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_messages,
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_advisor_with_router",
        new_callable=AsyncMock,
        return_value=advisor_resp,
    ):
        h = AdvisorOrchestrationHandler()
        result = await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
            litellm_logging_obj=fake_logging_obj,
        )

    assert executor_call_count == 2
    # Final result is the terminating executor response
    assert result is final_executor_resp

    # Outer response gets its own msg_<uuid> id so the proxy UI logs the
    # orchestrated request as a distinct trace (not colliding with any
    # inner sub-call's request_id).
    assert isinstance(result["id"], str) and result["id"].startswith("msg_")
    assert result["id"] != "msg_executor_final"

    # Aggregated cost on the response
    assert result["_hidden_params"]["response_cost"] == pytest.approx(2.0)

    # usage.iterations[] reports per-call breakdowns
    usage = result.get("usage")
    assert usage is not None
    iterations = usage.get("iterations")
    assert iterations is not None and len(iterations) == 3

    assert iterations[0]["type"] == "message"
    assert iterations[0]["input_tokens"] == 100
    assert iterations[0]["output_tokens"] == 20

    assert iterations[1]["type"] == "advisor_message"
    assert iterations[1]["model"] == "claude-opus-4-6"
    assert iterations[1]["input_tokens"] == 200
    assert iterations[1]["output_tokens"] == 150

    assert iterations[2]["type"] == "message"
    assert iterations[2]["input_tokens"] == 300
    assert iterations[2]["output_tokens"] == 40

    # Aggregated usage totals
    assert usage["input_tokens"] == 100 + 200 + 300
    assert usage["output_tokens"] == 20 + 150 + 40

    # cost_breakdown surfaced on the logging object
    fake_logging_obj.set_cost_breakdown.assert_called_once()
    breakdown_kwargs = fake_logging_obj.set_cost_breakdown.call_args.kwargs
    assert breakdown_kwargs["total_cost"] == pytest.approx(2.0)
    assert breakdown_kwargs["input_cost"] == pytest.approx(0.7)
    assert breakdown_kwargs["additional_costs"] == {
        "Main Model (initial)": pytest.approx(1.0),
        "Advisor Model": pytest.approx(0.3),
    }

    # Aggregate cost also recorded on logging model_call_details for downstream
    # transforms that drop the dict ``_hidden_params``.
    assert fake_logging_obj.model_call_details["response_cost"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_advisor_orchestration_no_advisor_call_no_additional_costs():
    """
    Executor produces the final response on first try (no advisor call).
    - ``Main Model (initial)`` and ``Advisor Model`` should not be present
      in ``additional_costs`` (both zero → dict is empty/None).
    - ``usage.iterations`` has exactly one entry.
    - Total cost matches the single executor turn.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    final_resp = _make_final_executor_response(response_cost=0.5)

    fake_logging_obj = MagicMock()
    fake_logging_obj.model_call_details = {}
    fake_logging_obj.set_cost_breakdown = MagicMock()

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        new_callable=AsyncMock,
        return_value=final_resp,
    ):
        h = AdvisorOrchestrationHandler()
        result = await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
            litellm_logging_obj=fake_logging_obj,
        )

    assert result["_hidden_params"]["response_cost"] == pytest.approx(0.5)

    # Even when no advisor is called, the orchestrator replaces the inner
    # sub-call id with a fresh msg_<uuid> so the UI log row is distinct.
    assert isinstance(result["id"], str) and result["id"].startswith("msg_")
    assert result["id"] != "msg_executor_final"

    iterations = result["usage"]["iterations"]
    assert len(iterations) == 1
    assert iterations[0]["type"] == "message"

    fake_logging_obj.set_cost_breakdown.assert_called_once()
    breakdown_kwargs = fake_logging_obj.set_cost_breakdown.call_args.kwargs
    assert breakdown_kwargs["total_cost"] == pytest.approx(0.5)
    assert breakdown_kwargs["input_cost"] == pytest.approx(0.5)
    assert breakdown_kwargs["additional_costs"] in (None, {})
