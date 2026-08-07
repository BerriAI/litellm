"""Live e2e: DataDog log delivery for calls that FAIL at the provider.

Covers logging.datadog.failure.exports_metric: when a call dies upstream the
caller gets an error and no usage, so the DataDog event is the only record that
the request ever happened. An operator's failure-rate alert is built on it, so
the event has to arrive, has to say the call failed, and has to name what went
wrong - a success-only logging path, or one that swallows failures, silently
zeroes out that alert.

The failure is real, not simulated: the test registers a deployment whose
upstream api_key is invalid, so OpenAI itself rejects the call and litellm's
failure path - not its success path - is what has to deliver. Both halves of the
contract are asserted: the recorded state (the proxy reports a datadog callback
registered for failure events) and the enforced behavior (the event at the real
DataDog intake, carrying the provider's error class, code and provider name,
cross-checked against the 401 the caller received, and carrying the prompt of
the very call that failed).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from pydantic import BaseModel, ConfigDict

from callback_config import callback_enabled, registered_callbacks
from datadog_reader import DdLogEvent, DdLogsReader
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from logging_client import INVALID_UPSTREAM_API_KEY, LoggingClient
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

#: The integration's name in litellm's callback settings.
DD_CALLBACK_NAME = "datadog"
#: A deployment litellm routes to OpenAI, so the invalid key is rejected by
#: OpenAI rather than by litellm's own request validation.
UPSTREAM_MODEL = "openai/gpt-4o-mini"
#: Present only in OpenAI's own rejection, never in the gateway's auth error, so
#: it tells an upstream 401 (the behavior under test) apart from a proxy 401.
UPSTREAM_REJECTION = "OpenAIException"


class _DdErrorInformation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error_class: str
    error_code: str
    llm_provider: str


class _DdMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class _DdFailurePayload(BaseModel):
    """The fields of a failed call's StandardLoggingPayload the scenario pins."""

    model_config = ConfigDict(extra="ignore")

    litellm_call_id: str
    status: str
    model_group: str
    call_type: str
    response_cost: float
    total_tokens: int
    error_str: str
    error_information: _DdErrorInformation
    messages: list[_DdMessage] = []


@pytest.fixture
def datadog_failure_logging(client: LoggingClient) -> Iterator[None]:
    """The proxy must ship failed calls to DataDog for the duration of the test.
    A deployment that already does is left exactly as it is."""
    yield from callback_enabled(client, DD_CALLBACK_NAME, events=("failure",))


def _first_upstream_rejection(client: LoggingClient, key: str, model: str, prompt: str) -> StreamingResponse:
    """The first response that is the provider's rejection rather than the
    gateway's own. A freshly created key can briefly 401 at the gateway until the
    data plane's auth cache picks it up, and that 401 never reaches the provider,
    so it would leave nothing for the integration to deliver; retry to a deadline
    until the body carries the upstream error.

    A gateway 401 is itself logged as a failure (ProxyLogging.post_call_failure_hook
    routes an auth_error on an llm api route into async_failure_handler), and that
    event carries the same prompt, so a discarded attempt is indistinguishable from
    the accepted one by prompt alone. The caller must therefore correlate on the
    returned response's x-litellm-call-id, which is unique per attempt."""
    deadline = time.monotonic() + client.proxy.poll_timeout
    while True:
        outcome = client.chat_raw(key, model, prompt, max_tokens=16)
        if UPSTREAM_REJECTION in outcome.body:
            return outcome
        if time.monotonic() >= deadline:
            pytest.fail(
                f"the call never reached the provider (status {outcome.status_code}): {outcome.body[:300]}"
            )
        time.sleep(client.proxy.poll_interval)


class TestDataDogFailureLogDelivery:
    @pytest.mark.covers("logging.datadog.failure.exports_metric", exercised_on=["chat_completions"])
    def test_chat_completions_failure_emits_one_log_event(
        self,
        client: LoggingClient,
        dd_logs: DdLogsReader,
        resources: ResourceManager,
        datadog_failure_logging: None,
    ) -> None:
        """A /chat/completions call the provider rejects must reach the DataDog
        logs intake as exactly one log event whose payload records the failure,
        names the upstream error, and carries the prompt of the failed call."""
        assert DD_CALLBACK_NAME in registered_callbacks(client, "failure"), (
            "the proxy must report the datadog callback registered for failure events "
            "before delivery can be asserted"
        )

        marker = unique_marker()
        model_name = f"dd-failure-{marker}"
        model_id = client.create_model(
            model_name, LiteLLMParamsBody(model=UPSTREAM_MODEL, api_key=INVALID_UPSTREAM_API_KEY)
        )
        resources.defer(lambda: client.delete_model(model_id))

        key = client.key_with_alias(f"dd-failure-{marker}", models=[model_name])
        resources.defer(lambda: client.delete_key(key))

        prompt = f"reply with one word {marker}"
        outcome = _first_upstream_rejection(client, key, model_name, prompt)
        assert outcome.status_code == 401, (
            f"an invalid upstream key must surface as the provider's 401, got {outcome.status_code}: "
            f"{outcome.body[:300]}"
        )
        call_id = outcome.call_id
        assert call_id is not None, "the failed response must still carry x-litellm-call-id"

        events = dd_logs.poll_events_for_marker(call_id)
        assert events, "no DataDog log event for the failed call reached the intake within the deadline"
        assert len(events) == 1, (
            f"expected exactly ONE DataDog log event for call {call_id}, got {len(events)}"
        )
        event: DdLogEvent = events[0]
        assert "source:litellm" in event.tags, (
            f"the ingested event must carry the litellm source (shipped as ddsource), got tags {event.tags!r}"
        )
        assert event.status != "ok", (
            "the integration ships a failure at DataDogStatus.ERROR, so the ingested event must not "
            f"index at DataDog's ok severity - which is where a failure logged as a success lands; got "
            f"{event.status!r}"
        )

        payload = _DdFailurePayload.model_validate(event.attributes)
        assert payload.litellm_call_id == call_id, (
            f"the event must be the one for the attempt the caller accepted; searched {call_id}, "
            f"got a payload for {payload.litellm_call_id}"
        )
        assert payload.status == "failure", f"payload status must be failure, got {payload.status!r}"
        assert payload.model_group == model_name, (
            f"payload model_group must be {model_name!r}, got {payload.model_group!r}"
        )
        assert payload.call_type == "acompletion", (
            f"payload call_type must be acompletion, got {payload.call_type!r}"
        )
        assert payload.error_information.error_code == "401", (
            f"the payload must carry the provider's status code, got {payload.error_information.error_code!r}"
        )
        assert payload.error_information.error_class == "AuthenticationError", (
            f"the payload must name the upstream error class, got {payload.error_information.error_class!r}"
        )
        assert payload.error_information.llm_provider == "openai", (
            f"the payload must name the provider that rejected the call, got "
            f"{payload.error_information.llm_provider!r}"
        )
        assert UPSTREAM_REJECTION in payload.error_str, (
            f"the payload's error_str must carry the provider's own message, got {payload.error_str!r}"
        )
        assert payload.response_cost == 0, (
            "a failed call bills nothing, so a non-zero cost means the failure was accounted as if it "
            f"had succeeded; got {payload.response_cost}"
        )
        assert payload.total_tokens == 0, (
            f"a failed call consumes no tokens, got {payload.total_tokens}"
        )
        assert any(marker in message.content for message in payload.messages), (
            f"the delivered event must carry the failed call's own prompt, got {payload.messages!r}"
        )
