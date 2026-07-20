"""Live e2e: POST /v1/messages (Anthropic Messages API) product contract.

Registers an Anthropic deployment at runtime. Asserts assistant text, SSE
streaming, x-litellm-call-id, and cost (response header + SpendLogs) so Messages
is not a single nonstream smoke check.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, MessagesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND = "anthropic/claude-haiku-4-5"


def _provision(
    endpoints_client: EndpointsClient, resources: ResourceManager, prefix: str
) -> str:
    model = f"{prefix}-{unique_marker()}"
    model_id = endpoints_client.create_model(
        model,
        LiteLLMParamsBody(
            model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"
        ),
    )
    resources.defer(lambda: endpoints_client.delete_model(model_id))
    return model


class TestAnthropicMessages:
    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.works")
    def test_messages_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _provision(endpoints_client, resources, "e2e-messages")
        key = resources.key()

        result = endpoints_client.messages(key, model, "reply with one word")
        require_successful_call(result)
        assert result.call_id, f"/v1/messages must set x-litellm-call-id; got {result!r}"
        parsed = MessagesResult.model_validate_json(result.body)
        assert parsed.role == "assistant", f"unexpected role: {result.body[:300]}"
        assert parsed.text.strip(), f"/v1/messages returned no text: {result.body[:300]}"

    @pytest.mark.covers("llm.messages.anthropic.basic.nonstream.cost_logged")
    def test_messages_logs_cost(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _provision(endpoints_client, resources, "e2e-messages-cost")
        key = resources.key()

        result = endpoints_client.messages(
            key, model, f"reply with one word {unique_marker()}"
        )
        require_successful_call(result)
        assert result.call_id
        assert result.response_cost is not None and result.response_cost > 0, (
            f"nonstream /v1/messages must set x-litellm-response-cost > 0; "
            f"got {result.response_cost!r}"
        )
        rows = endpoints_client.proxy.poll_logs_for_key(
            key,
            min_rows=1,
            predicate=lambda logged_rows: any((row.spend or 0) > 0 for row in logged_rows),
        )
        assert rows, "no costed SpendLogs row for messages call"
        assert (rows[0].spend or 0) > 0

    @pytest.mark.covers("llm.messages.anthropic.basic.stream.works")
    def test_messages_streaming_returns_events(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _provision(endpoints_client, resources, "e2e-messages-stream")
        key = resources.key()

        result = endpoints_client.messages(
            key, model, f"reply with one word {unique_marker()}", stream=True
        )
        require_successful_call(result)
        assert result.call_id, "streamed /v1/messages must set x-litellm-call-id"
        assert result.is_streaming, (
            f"streamed /v1/messages must be text/event-stream, got {result.content_type!r}"
        )
        assert result.chunks > 0, "streamed /v1/messages produced no SSE events"
        assert result.stream_error is None, (
            f"stream carried an upstream error after HTTP 200: {result.stream_error}"
        )

    @pytest.mark.covers(
        "llm.messages.anthropic.basic.stream.works",
        "llm.messages.anthropic.basic.nonstream.cost_logged",
    )
    def test_messages_stream_logs_spend(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _provision(endpoints_client, resources, "e2e-messages-stream-cost")
        key = resources.key()

        result = endpoints_client.messages(
            key, model, f"reply with one word {unique_marker()}", stream=True
        )
        require_successful_call(result)
        assert result.chunks > 0
        rows = endpoints_client.proxy.poll_logs_for_key(
            key,
            min_rows=1,
            predicate=lambda logged_rows: any((row.spend or 0) > 0 for row in logged_rows),
        )
        assert rows, "streamed /v1/messages must land a costed SpendLogs row"
        assert (rows[0].spend or 0) > 0, f"stream spend not costed: {rows[0]}"
