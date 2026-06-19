"""Live e2e for LLM-translation passthrough endpoints.

Each test sends a NATIVE provider request through the proxy's passthrough route
and verifies the proxy still logged a costed SpendLogs row
(call_type="pass_through_endpoint"), correlated by the x-litellm-call-id header.

Covered: gemini ("gemini-2.5-flash") + anthropic ("claude-haiku-4-5"), streaming +
non-streaming, plus native tool calls. See LLM_TRANSLATION_COVERAGE_MATRIX.md.

A passthrough call returning non-2xx fails hard (never a skip); once it returns
2xx, a missing or zero-cost SpendLogs row fails too.
"""

import pytest

from passthrough_client import PassthroughClient, PassthroughResult
from proxy_client import SpendLogRow, require_successful_call, unique_marker

pytestmark = pytest.mark.e2e


def _f(value: object) -> float:
    return float(value) if isinstance(value, (int, float, str)) else 0.0


def _s(value: object) -> str:
    return str(value) if value is not None else ""


def _costed_row(client: PassthroughClient, result: PassthroughResult) -> SpendLogRow:
    """The passthrough call's logged row, polled until it carries a cost.

    Asserts (not skips) that a 2xx passthrough call produced a costed row - the
    whole point of passthrough spend tracking.
    """
    assert result.call_id, "passthrough response had no x-litellm-call-id header"
    rows = client.poll_logs_for_request_id(
        result.call_id,
        predicate=lambda rs: _f(rs[0].get("spend")) > 0,
    )
    assert rows, f"no SpendLogs row for passthrough call_id {result.call_id}"
    row = rows[0]
    assert _s(row.get("call_type")) == "pass_through_endpoint"
    assert _f(row.get("spend")) > 0, f"passthrough call was not costed: {row}"
    assert _s(row.get("status")) == "success"
    return row


# ---- Gemini passthrough ------------------------------------------------


def test_gemini_passthrough_nonstreaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    tag = f"e2e-passthrough-{unique_marker()}"
    result = client.gemini_generate(
        scoped_key, "gemini-2.5-flash", "Say hello in one word", tags=[tag, "gemini"]
    )
    require_successful_call(result)

    row = _costed_row(client, result)
    assert _s(row.get("custom_llm_provider")) == "gemini"
    assert "gemini" in _s(row.get("model"))
    assert tag in _s(row.get("request_tags")), f"tags not logged: {row.get('request_tags')}"


def test_gemini_passthrough_streaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.gemini_stream(scoped_key, "gemini-2.5-flash", "Count to five")
    require_successful_call(result)
    assert result.chunks > 0, "streaming passthrough produced no events"

    row = _costed_row(client, result)
    assert _s(row.get("custom_llm_provider")) == "gemini"


def test_gemini_passthrough_tool_call_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.gemini_generate(
        scoped_key,
        "gemini-2.5-flash",
        "What is the weather in Paris? Use the get_weather tool.",
        tools=[
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ]
            }
        ],
    )
    require_successful_call(result)
    assert "functionCall" in result.body, "gemini did not emit a tool call"

    row = _costed_row(client, result)
    assert _s(row.get("custom_llm_provider")) == "gemini"


# ---- Anthropic passthrough ---------------------------------------------


def test_anthropic_passthrough_nonstreaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(scoped_key, "claude-haiku-4-5", "Say hello")
    require_successful_call(result)

    row = _costed_row(client, result)
    assert _s(row.get("custom_llm_provider")) == "anthropic"
    assert "claude" in _s(row.get("model"))


def test_anthropic_passthrough_streaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(
        scoped_key, "claude-haiku-4-5", "Count to five", stream=True
    )
    require_successful_call(result)
    assert result.chunks > 0, "streaming passthrough produced no events"

    row = _costed_row(client, result)
    assert _s(row.get("custom_llm_provider")) == "anthropic"


def test_anthropic_passthrough_tool_call_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(
        scoped_key,
        "claude-haiku-4-5",
        "What is the weather in Paris? Use the get_weather tool.",
        tools=[
            {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    )
    require_successful_call(result)
    assert "tool_use" in result.body, "anthropic did not emit a tool call"

    row = _costed_row(client, result)
    assert _s(row.get("custom_llm_provider")) == "anthropic"
