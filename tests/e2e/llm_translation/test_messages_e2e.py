"""Live e2e: POST /v1/messages (Anthropic Messages API) returns a real completion.

Registers an Anthropic deployment at runtime, drives the Messages endpoint through
the gateway, and asserts an assistant message with text came back. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, MessagesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, SpendLogRow

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND = "anthropic/claude-haiku-4-5"


def _approx_equal(actual: float, expected: float) -> bool:
    """Within 1% or 1e-9 absolute - spend math, not exact float identity."""
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


class TestAnthropicMessages:
    def test_messages_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-messages-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.messages(key, model, "reply with one word")
        require_successful_call(result)
        parsed = MessagesResult.model_validate_json(result.body)
        assert parsed.role == "assistant", f"unexpected role: {result.body[:300]}"
        assert parsed.text.strip(), f"/v1/messages returned no text: {result.body[:300]}"

    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.cost_logged")
    def test_messages_logs_cost_matching_the_response_header(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        require_env("ANTHROPIC_API_KEY")
        model = f"e2e-messages-cost-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.messages(key, model, f"reply with one word {unique_marker()}")
        require_successful_call(result)
        parsed = MessagesResult.model_validate_json(result.body)
        assert parsed.role == "assistant" and parsed.text.strip(), (
            f"/v1/messages returned no assistant text: {result.body[:300]}"
        )
        assert parsed.id, f"/v1/messages returned no id to correlate the spend row: {result.body[:300]}"

        # The customer reads per-request cost off the response header (LIT-4076), so
        # it must be present and positive on /v1/messages, not only /chat/completions.
        header_cost = result.response_cost
        assert header_cost is not None and header_cost > 0, (
            "x-litellm-response-cost header missing or non-positive on /v1/messages; "
            f"headers={result.headers}"
        )

        # SpendLogs.request_id is the completion body id, so correlate by the message
        # id and wait for the row's cost to flush (proxy_batch_write_at, ~60s).
        def _priced(rows: list[SpendLogRow]) -> bool:
            return any(r.spend is not None and r.spend > 0 for r in rows)

        rows = endpoints_client.proxy.poll_logs_for_request_id(parsed.id, predicate=_priced)
        priced = [r for r in rows if r.request_id == parsed.id and r.spend is not None and r.spend > 0]
        assert priced, (
            f"no priced /spend/logs row landed for message {parsed.id} within the poll window; got {rows}"
        )
        row = priced[0]
        assert (row.prompt_tokens or 0) > 0 and (row.completion_tokens or 0) > 0, (
            f"messages spend row missing token counts, so the cost is not real usage: {row}"
        )
        assert row.spend is not None and _approx_equal(row.spend, header_cost), (
            f"logged spend {row.spend} disagrees with the x-litellm-response-cost header {header_cost}; "
            "the customer bills against the header, so the two must match"
        )
