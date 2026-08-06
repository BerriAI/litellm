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

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from models import KeyGenerateBody, SpendLogRow
from passthrough_client import (
    AnthropicTool,
    GeminiFunctionDeclaration,
    GeminiTool,
    JsonSchema,
    JsonSchemaProperty,
    PassthroughClient,
)

pytestmark = pytest.mark.e2e


def _fetch_cost_breakdown(client: PassthroughClient, result: StreamingResponse) -> SpendLogRow:
    """The passthrough call's logged row, polled until it carries a cost.

    Asserts (not skips) that a 2xx passthrough call produced a costed row - the
    whole point of passthrough spend tracking.
    """
    assert result.call_id, "passthrough response had no x-litellm-call-id header"
    rows = client.proxy.poll_logs_for_request_id(
        result.call_id,
        predicate=lambda rs: (rs[0].spend or 0) > 0,
    )
    assert rows, f"no SpendLogs row for passthrough call_id {result.call_id}"
    row = rows[0]
    assert row.call_type == "pass_through_endpoint"
    assert (row.spend or 0) > 0, f"passthrough call was not costed: {row}"
    assert row.status == "success"
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

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "gemini"
    assert "gemini" in (row.model or "")
    assert tag in (row.request_tags or []), f"tags not logged: {row.request_tags}"


def test_gemini_passthrough_streaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.gemini_stream(scoped_key, "gemini-2.5-flash", "Count to five")
    require_successful_call(result)
    assert result.chunks > 0, "streaming passthrough produced no events"

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "gemini"


def test_gemini_passthrough_tool_call_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.gemini_generate(
        scoped_key,
        "gemini-2.5-flash",
        "What is the weather in Paris? Use the get_weather tool.",
        tools=[
            GeminiTool(
                function_declarations=[
                    GeminiFunctionDeclaration(
                        name="get_weather",
                        description="Get the weather for a city",
                        parameters=JsonSchema(
                            type="object",
                            properties={"city": JsonSchemaProperty(type="string")},
                            required=["city"],
                        ),
                    )
                ]
            )
        ],
    )
    require_successful_call(result)
    assert "functionCall" in result.body, "gemini did not emit a tool call"

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "gemini"


# ---- Anthropic passthrough ---------------------------------------------


def test_anthropic_passthrough_nonstreaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(scoped_key, "claude-haiku-4-5", "Say hello")
    require_successful_call(result)

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "anthropic"
    assert "claude" in (row.model or "")


def test_anthropic_passthrough_streaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(
        scoped_key, "claude-haiku-4-5", "Count to five", stream=True
    )
    require_successful_call(result)
    assert result.chunks > 0, "streaming passthrough produced no events"

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "anthropic"


def test_anthropic_passthrough_tool_call_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(
        scoped_key,
        "claude-haiku-4-5",
        "What is the weather in Paris? Use the get_weather tool.",
        tools=[
            AnthropicTool(
                name="get_weather",
                description="Get the weather for a city",
                input_schema=JsonSchema(
                    type="object",
                    properties={"city": JsonSchemaProperty(type="string")},
                    required=["city"],
                ),
            )
        ],
    )
    require_successful_call(result)
    assert "tool_use" in result.body, "anthropic did not emit a tool call"

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "anthropic"


class TestPassthroughModelAllowlist:
    """A passthrough route must honor the calling key's model allow-list.

    The customer fronts native provider calls through the proxy with custom auth,
    so a key scoped to one model must not reach a different model just because the
    request goes through the passthrough route rather than /chat/completions.
    """

    @pytest.mark.covers("other.auth.passthrough.model_allowlist_enforced")
    def test_passthrough_denies_model_outside_key_allowlist(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        key = client.proxy.generate_key(KeyGenerateBody(models=["gemini-2.5-flash"]))
        resources.defer(lambda: client.proxy.delete_key(key))

        result = client.anthropic_message(key, "claude-haiku-4-5", f"say hi {unique_marker()}")
        assert result.status_code == 403, (
            "a key restricted to gemini-2.5-flash must be denied a claude passthrough call, "
            f"got {result.status_code}: {result.body[:300]}"
        )
