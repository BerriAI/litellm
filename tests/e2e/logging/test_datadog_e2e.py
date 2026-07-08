"""Live e2e: the proxy's ``datadog`` callback ships every request to Datadog.

The proxy is configured with ``datadog`` in its callbacks, so each completion
(success or failure) has its StandardLoggingPayload batched and flushed to the
Datadog logs intake. These tests drive real traffic through the proxy, then read
the events back out of the Datadog Logs Search API to prove they actually landed -
delivery is verified at the destination, not by trusting the proxy.

Each test stamps a unique marker into the prompts it sends. That marker rides
along in the logged payload's ``messages``, so a free-text search for it isolates
exactly this run's events from every other request flowing through the shared
proxy (and across concurrent CI runs). Delivery is asynchronous - the proxy
batches and flushes (every 5s or at ``DD_MAX_BATCH_SIZE``) and Datadog then
indexes for search - so the read-back polls to a deadline instead of reading once.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import is_ok, require_successful_call
from lifecycle import ResourceManager
from logging_client import DD_ERROR, DatadogClient, LoggingClient
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

MODEL = "gpt-5.5"
ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5"
BATCH_REQUESTS = 10
RESPONSES_REQUESTS = 5


def _probe_prompt(run_marker: str, index: int) -> str:
    """Unique per request so the cache never collapses two into one, but every
    prompt carries ``run_marker`` so a single search finds the whole batch."""
    return f"e2e datadog probe {run_marker} item {index} {unique_marker()}: reply with OK"


class TestDatadogLogging:
    @pytest.mark.covers("logging.datadog.success.exports_metric", exercised_on=["chat_completions"])
    def test_datadog_can_flush_logs(
        self, client: LoggingClient, resources: ResourceManager, datadog: DatadogClient
    ) -> None:
        """A batch of successful completions is delivered to Datadog with no drops:
        driving N requests must produce at least N searchable log events carrying
        this run's marker. A callback that stops shipping, drops on flush, or loses
        the payload's message content fails here."""
        run_marker = f"e2edd{unique_marker()}"
        key = client.key_with_alias(f"e2e-dd-{run_marker}", models=[MODEL])
        resources.defer(lambda: client.delete_key(key))

        for index in range(BATCH_REQUESTS):
            response = client.chat(key, MODEL, _probe_prompt(run_marker, index))
            assert response.choices, f"empty completion for {run_marker} item {index}: {response}"

        events = datadog.poll_for_events(run_marker, min_count=BATCH_REQUESTS)
        assert len(events) >= BATCH_REQUESTS, (
            f"Datadog returned {len(events)} events for marker {run_marker}, expected >= {BATCH_REQUESTS}; "
            "the proxy's datadog callback dropped success logs or stopped shipping them"
        )

    @pytest.mark.covers("logging.datadog.success.exports_metric", exercised_on=["responses"])
    def test_datadog_logs_responses_api_with_claude(
        self, client: LoggingClient, resources: ResourceManager, datadog: DatadogClient
    ) -> None:
        """The Responses API on a Claude deployment is delivered to Datadog too:
        the same no-drop contract as chat, but exercising litellm's
        Responses-to-Anthropic translation and the /v1/responses logging path. A
        callback that only ships /chat/completions would fail here."""
        run_marker = f"e2eddresp{unique_marker()}"
        model = f"dd-responses-{run_marker}"
        model_id = client.gateway.create_model(
            model,
            LiteLLMParamsBody(model=ANTHROPIC_MODEL, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
        resources.defer(lambda: client.gateway.delete_model(model_id))
        key = client.key_with_alias(f"e2e-ddr-{run_marker}", models=[model])
        resources.defer(lambda: client.delete_key(key))

        for index in range(RESPONSES_REQUESTS):
            result = client.responses(key, model, _probe_prompt(run_marker, index))
            require_successful_call(result)

        events = datadog.poll_for_events(run_marker, min_count=RESPONSES_REQUESTS)
        assert len(events) >= RESPONSES_REQUESTS, (
            f"Datadog returned {len(events)} events for responses marker {run_marker}, "
            f"expected >= {RESPONSES_REQUESTS}; the proxy's datadog callback did not ship the "
            "/v1/responses logs"
        )

    @pytest.mark.covers("logging.datadog.failure.exports_metric", exercised_on=["chat_completions"])
    def test_datadog_logs_request_failures(
        self, client: LoggingClient, resources: ResourceManager, datadog: DatadogClient
    ) -> None:
        """A failed completion is delivered to Datadog and classified as an error.
        A deployment wired to a bad upstream key makes the provider reject the call;
        the resulting failure must surface as a Datadog event tagged ``error`` that
        still carries this run's marker."""
        run_marker = f"e2eddfail{unique_marker()}"
        bad_model = f"dd-fail-{run_marker}"
        model_id = client.gateway.create_model(
            bad_model,
            LiteLLMParamsBody(model="openai/gpt-5.5", api_key="sk-invalid-e2e-datadog"),
        )
        resources.defer(lambda: client.gateway.delete_model(model_id))
        key = client.key_with_alias(f"e2e-ddf-{run_marker}", models=[bad_model])
        resources.defer(lambda: client.delete_key(key))

        result = client.chat_result(key, bad_model, f"e2e datadog failure probe {run_marker}: reply with OK")
        assert not is_ok(result), f"expected the bad-key deployment to reject the call, got {result}"

        events = datadog.poll_for_events(run_marker, min_count=1)
        assert events, (
            f"Datadog returned no events for failed-request marker {run_marker}; "
            "the proxy's datadog callback did not ship the failure"
        )
        statuses = [event.attributes.status for event in events if event.attributes]
        assert DD_ERROR in statuses, (
            f"failed request for {run_marker} was shipped to Datadog but not as an error (statuses seen: {statuses})"
        )
