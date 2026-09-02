"""Live e2e: key-scoped Weave (Weights & Biases) delivery, success and failure.

Covers the two `logging.niche_integrations.*.logs_spend` cells with a real member
of that cohort. A key carrying a `weave_otel` callback in its logging metadata
must deliver its calls to the real Weave project, and each call must arrive
exactly once, carrying the same cost the response header reported:

- success: one `litellm_request` call, OTEL status OK, `llm.response.cost` equal
  to `x-litellm-response-cost`, and non-zero tokens
- failure: a provider-rejected call arrives too, as one call with OTEL status
  ERROR naming the provider exception, and with no cost - a failed call that
  silently never reaches the destination is an invisible outage, and a billed
  one is worse

Both halves assert the recorded state (the key's callback registration answers
success and the destination holds the call) and the enforced behavior (the
delivered payload's status and cost). Delivery is read back through Weave's own
query API; nothing is mocked.
"""

from __future__ import annotations

import time

import pytest

from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from logging_client import (
    INVALID_UPSTREAM_API_KEY,
    LoggingClient,
    WeaveCreds,
    costs_agree,
    first_ok,
    load_weave_creds,
)
from models import LiteLLMParamsBody
from weave_reader import WeaveCall, WeaveReader, build_weave_reader

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="session")
def weave_creds() -> WeaveCreds:
    return load_weave_creds()


@pytest.fixture(scope="session")
def weave_reader() -> WeaveReader:
    return build_weave_reader()


#: How far before the request the Weave read-back window opens, to absorb clock
#: skew between this host and Weave. Without it a host running slightly fast
#: would filter out its own call.
_WINDOW_SKEW_SECONDS = 120.0


def _window_start() -> float:
    return time.time() - _WINDOW_SKEW_SECONDS


def _exactly_one(calls: tuple[WeaveCall, ...], *, marker: str, what: str) -> WeaveCall:
    assert calls, f"no Weave call for the {what} (marker {marker}) reached the project within the deadline"
    assert len(calls) == 1, (
        f"expected exactly ONE Weave call for the {what} (marker {marker}), got {len(calls)}: "
        f"{[call.id for call in calls]} - more than one call for one request is the "
        "duplicate-delivery bug"
    )
    return calls[0]


class TestWeaveLogDelivery:
    @pytest.mark.covers("logging.niche_integrations.success.logs_spend", exercised_on=["chat_completions"])
    def test_chat_completions_delivers_one_call_with_spend(
        self,
        client: LoggingClient,
        weave_creds: WeaveCreds,
        weave_reader: WeaveReader,
        resources: ResourceManager,
    ) -> None:
        alias = f"weave-key-{unique_marker()}"
        key = client.key_with_alias(
            alias,
            models=[CHEAP_ANTHROPIC_MODEL],
            metadata=weave_creds.key_logging_metadata(),
        )
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        since = _window_start()
        outcome = first_ok(
            client,
            lambda: client.chat_raw(key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", max_tokens=64),
        )
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"the response must report x-litellm-response-cost, got {outcome.response_cost!r}"
        )

        call = _exactly_one(
            weave_reader.poll_calls_matching(marker, since=since), marker=marker, what="successful call"
        )

        assert call.status_code == "OK", f"a successful call must land at OK span status, got {call.status_code!r}"
        cost = call.response_cost
        assert cost is not None and costs_agree(outcome.response_cost, cost), (
            f"the Weave call's llm.response.cost {cost!r} must agree with the header cost "
            f"{outcome.response_cost} - a delivered span with the wrong cost is a silent "
            "billing-attribution bug"
        )
        assert call.total_tokens is not None and call.total_tokens > 0, (
            f"the delivered call must carry token usage, got {call.total_tokens!r}"
        )

    @pytest.mark.covers("logging.niche_integrations.failure.logs_spend", exercised_on=["chat_completions"])
    def test_failed_chat_completions_delivers_one_error_call(
        self,
        client: LoggingClient,
        weave_creds: WeaveCreds,
        weave_reader: WeaveReader,
        resources: ResourceManager,
    ) -> None:
        """A deployment with an invalid upstream key passes proxy auth and fails
        at the provider, so exactly one provider failure exists for it. Proxy-side
        401s during key propagation never reach the provider and ship no payload,
        which is what the retry loop below relies on."""
        model_name = f"weave-err-{unique_marker()}"
        model_id = client.create_model(
            model_name,
            LiteLLMParamsBody(model="anthropic/claude-haiku-4-5", api_key=INVALID_UPSTREAM_API_KEY),
        )
        resources.defer(lambda: client.delete_model(model_id))
        key = client.key_with_alias(
            f"weave-err-key-{unique_marker()}",
            models=[model_name],
            metadata=weave_creds.key_logging_metadata(),
        )
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        since = _window_start()
        outcome = _provoke_provider_failure(client, key, model_name, marker)

        call = _exactly_one(weave_reader.poll_calls_matching(marker, since=since), marker=marker, what="failed call")

        assert call.status_code == "ERROR", (
            f"a failed call must land at ERROR span status, got {call.status_code!r} - "
            "Weave's own summary.weave.status reads success either way, which is exactly "
            "why the span status is what this asserts on"
        )
        error = call.error
        assert error is not None and error.message is not None and "AnthropicException" in error.message, (
            f"the delivered call must carry the provider error, got {error!r}"
        )
        assert not call.response_cost, f"a failed call must not be billed, got llm.response.cost={call.response_cost!r}"
        assert outcome.status_code == 401, (
            f"an upstream auth failure must map to 401, got {outcome.status_code}: {outcome.body[:200]}"
        )


def _provoke_provider_failure(client: LoggingClient, key: str, model_name: str, marker: str) -> StreamingResponse:
    """Send until the provider (not the proxy) is the one rejecting the call.

    A network failure between the test and the proxy is NOT retried: the request
    may have been served, and a retry would double-log the failure payload and
    falsely trip the exactly-one assertion.
    """
    deadline = time.monotonic() + client.proxy.poll_timeout
    while True:
        outcome = client.chat_raw(key, model_name, f"trigger an upstream auth failure {marker}", max_tokens=16)
        assert not outcome.ok, "the call must fail; the deployment's upstream key is invalid"
        assert outcome.status_code != -1, (
            "network failure between the test and the proxy while provoking the provider failure; "
            "retrying now could double-log the failure payload and falsely trip the exactly-one "
            f"assertion - fix the rig connectivity first: {outcome.body[:200]}"
        )
        if "AnthropicException" in outcome.body or time.monotonic() >= deadline:
            break
        time.sleep(client.proxy.poll_interval)
    assert "AnthropicException" in outcome.body, (
        "never saw the upstream provider failure before the deadline; the key may still be "
        f"propagating - last outcome {outcome.status_code}: {outcome.body[:200]}"
    )
    return outcome
