"""StandardLoggingPayload for failed requests, and how many payloads retries and fallbacks produce.

The rig runs `callbacks: ["generic_api", <custom recorder>]` with `num_retries: 0` and cooldowns off, against a
scripted provider whose deployments answer 400, 401, 429 or 500 on purpose. A failed request must still produce a
payload: status "failure", zero tokens and cost, the request that was sent, and an `error_information` object that
names the HTTP status, the litellm exception class, the provider and the provider's message. Failures the proxy
raises before any provider call (bad key, model not allowed) are logged too. The last tests pin the count of
payloads for one request that is retried or falls back, which is what makes failure dashboards and per-attempt
billing add up.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import object_value
from integration.observability.dec1_callbacks_support import (
    CHAT,
    CHAT_COST,
    CHAT_DENIED,
    CHAT_FALLBACK_PRIMARY,
    CHAT_FLAKY,
    CHAT_LIMITED,
    CHAT_OUTAGE,
    CHAT_REJECTED,
    EMBEDDING_REJECTED,
    MESSAGES_REJECTED,
    PROVIDER_DENIAL,
    PROVIDER_KEY,
    PROVIDER_LIMIT,
    PROVIDER_OUTAGE,
    PROVIDER_REJECTION,
    RECORDER,
    RESPONSES_REJECTED,
    CallbackRig,
    callback_rig,
    chat,
    key_hash,
    marker,
    new_call_id,
    post,
    provider_bodies,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-payload-failures"), {"callbacks": ["generic_api", RECORDER]}
    ) as created:
        yield created


def test_provider_400_delivers_exactly_one_failure_payload(rig: CallbackRig) -> None:
    """The provider answers the chat request with a 400. The caller gets a 400, and the generic_api endpoint gets
    exactly one payload for the call id, with status "failure" and
    status_fields.llm_api_status "failure". No success payload is delivered."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, model=CHAT_REJECTED)
    assert response.status_code == 400, response.text
    event: Final = rig.settled_events(call_id, 1)[0]
    assert event["status"] == "failure"
    assert object_value(event["status_fields"])["llm_api_status"] == "failure"


@pytest.mark.parametrize(
    ("model", "status", "error_class", "message"),
    [
        pytest.param(CHAT_REJECTED, 400, "BadRequestError", PROVIDER_REJECTION, id="provider-400"),
        pytest.param(CHAT_DENIED, 401, "AuthenticationError", PROVIDER_DENIAL, id="provider-401"),
        pytest.param(CHAT_LIMITED, 429, "RateLimitError", PROVIDER_LIMIT, id="provider-429"),
        pytest.param(CHAT_OUTAGE, 500, "InternalServerError", PROVIDER_OUTAGE, id="provider-500"),
    ],
)
def test_error_information_names_status_class_provider_and_message(
    rig: CallbackRig, model: str, status: int, error_class: str, message: str
) -> None:
    """The provider answers with 400, 401, 429 or 500 and a JSON error message. The caller gets the same status.
    The failure payload's `error_information` must carry error_code as that status in string form, error_class as
    the litellm exception class (BadRequestError, AuthenticationError, RateLimitError, InternalServerError),
    llm_provider "openai", and an error_message that contains the provider's message. `error_str` must contain the
    provider's message too."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, model=model)
    assert response.status_code == status, response.text
    event: Final = rig.event(call_id, "failure")
    information: Final = object_value(event["error_information"])
    assert (information["error_code"], information["error_class"], information["llm_provider"]) == (
        str(status),
        error_class,
        "openai",
    )
    assert message in str(information["error_message"])
    assert message in str(event["error_str"])


def test_provider_429_is_categorised_as_a_vendor_rate_limit(rig: CallbackRig) -> None:
    """A 429 that came from the provider (not from the proxy's own limits) must log
    `error_information.error_rate_limit_category` "vendor_rate_limit", so dashboards can tell provider throttling
    from proxy throttling."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, model=CHAT_LIMITED).status_code == 429
    information: Final = object_value(rig.event(call_id, "failure")["error_information"])
    assert information["error_rate_limit_category"] == "vendor_rate_limit"


def test_failure_payload_has_zero_tokens_and_zero_cost(rig: CallbackRig) -> None:
    """A request the provider rejected with 400 used nothing: prompt_tokens, completion_tokens and total_tokens
    are 0, response_cost is 0, and the `x-litellm-response-cost` header, when present on the error response, is 0."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, model=CHAT_REJECTED)
    event: Final = rig.event(call_id, "failure")
    assert (event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) == (0, 0, 0)
    assert event["response_cost"] == 0
    assert float(response.headers.get("x-litellm-response-cost", "0")) == 0


def test_failure_payload_id_falls_back_to_the_call_id(rig: CallbackRig) -> None:
    """There is no provider response id on a failure, so the payload's `id` must equal its `litellm_call_id`, which
    is the `x-litellm-call-id` the caller sent and got back on the error response."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, model=CHAT_REJECTED)
    assert response.headers["x-litellm-call-id"] == call_id
    event: Final = rig.event(call_id, "failure")
    assert (event["id"], event["litellm_call_id"]) == (call_id, call_id)


def test_failure_payload_keeps_the_request_and_names_the_deployment(rig: CallbackRig) -> None:
    """The failure payload must still describe what was attempted: `messages` holds the caller's user text,
    `model_group` is the requested public model, `custom_llm_provider` is "openai", `api_base` is the failing
    deployment's base URL and `call_type` is "acompletion"."""
    call_id: Final = new_call_id()
    text: Final = marker()
    assert chat(rig, call_id, text, model=CHAT_REJECTED).status_code == 400
    event: Final = rig.event(call_id, "failure")
    assert text in json.dumps(event["messages"])
    assert (event["model_group"], event["custom_llm_provider"], event["call_type"]) == (
        CHAT_REJECTED,
        "openai",
        "acompletion",
    )
    assert event["api_base"] == rig.provider.url + "/rejected/v1"


def test_failure_payload_identifies_the_key_and_tags(rig: CallbackRig) -> None:
    """A virtual key with an alias sends a tagged request that the provider rejects. The failure payload must carry
    the key's SHA-256 hash and alias in `metadata` and the caller's tag in `request_tags`, the same as a success."""
    call_id: Final = new_call_id()
    alias, tag = "alias-" + marker(), marker()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias)
        assert chat(rig, call_id, model=CHAT_REJECTED, key=key, metadata={"tags": [tag]}).status_code == 400
        event: Final = rig.event(call_id, "failure")
        metadata: Final = object_value(event["metadata"])
        assert (metadata["user_api_key_hash"], metadata["user_api_key_alias"]) == (key_hash(key), alias)
        assert tag in json.dumps(event["request_tags"])


def test_failure_payload_never_contains_the_provider_key(rig: CallbackRig) -> None:
    """The failing deployment is configured with a provider api key. Neither that key nor the caller's virtual key
    may appear anywhere in the failure payload, including the traceback and error message fields."""
    call_id: Final = new_call_id()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert chat(rig, call_id, model=CHAT_OUTAGE, key=key).status_code == 500
        text: Final = json.dumps(rig.event(call_id, "failure"))
        assert PROVIDER_KEY not in text
        assert key not in text


@pytest.mark.parametrize(
    ("route", "body", "call_type"),
    [
        pytest.param(
            "/v1/embeddings", {"model": EMBEDDING_REJECTED, "input": "dec1 failure"}, "aembedding", id="embeddings"
        ),
        pytest.param(
            "/v1/messages",
            {"model": MESSAGES_REJECTED, "max_tokens": 16, "messages": [{"role": "user", "content": "dec1 failure"}]},
            "anthropic_messages",
            id="messages",
        ),
        pytest.param(
            "/v1/responses", {"model": RESPONSES_REJECTED, "input": "dec1 failure"}, "aresponses", id="responses"
        ),
    ],
)
def test_provider_400_on_other_surfaces_logs_one_failure_with_that_call_type(
    rig: CallbackRig, route: str, body: dict[str, JsonValue], call_type: str
) -> None:
    """The provider rejects an embeddings, /v1/messages or /v1/responses request with 400. The caller gets 400 and
    exactly one payload arrives with status "failure", the surface's own `call_type`, error_code "400" and the
    provider's message in error_message."""
    call_id: Final = new_call_id()
    response: Final = post(rig, route, body, call_id)
    assert response.status_code == 400, response.text
    event: Final = rig.settled_events(call_id, 1)[0]
    information: Final = object_value(event["error_information"])
    assert (event["status"], event["call_type"], information["error_code"]) == ("failure", call_type, "400")
    assert PROVIDER_REJECTION in str(information["error_message"])


def test_invalid_virtual_key_is_logged_as_a_401_failure_without_the_key(rig: CallbackRig) -> None:
    """A caller presents a virtual key the proxy does not know. The caller gets 401, no provider request is made,
    and one failure payload arrives with error_information.error_code "401". The rejected key string itself must
    not appear anywhere in the payload. Key auth fails before the endpoint reads `x-litellm-call-id`, so the
    payload is found by the request's unique text (`CallbackRig.event_mentioning`)."""
    call_id: Final = new_call_id()
    text: Final = marker()
    bad_key: Final = "sk-dec1-unknown-" + marker()
    rig.provider.drain()
    response: Final = chat(rig, call_id, text, key=bad_key)
    assert response.status_code == 401, response.text
    event: Final = rig.event_mentioning(text, "failure")
    assert object_value(event["error_information"])["error_code"] == "401"
    assert bad_key not in json.dumps(event)
    assert provider_bodies(rig) == ()


def test_model_not_allowed_for_the_key_is_logged_as_a_failure_with_the_key_identity(rig: CallbackRig) -> None:
    """A key restricted to the embedding model asks for the chat model. The caller gets 401 or 403, no provider
    request is made, and one failure payload arrives that identifies the key by hash and alias. The model check
    runs inside key auth, before the endpoint reads `x-litellm-call-id`, so the payload is found by the request's
    unique text. UNVERIFIED: the exact status (401 vs 403) and error_class were not traced; see the model access
    check in litellm/proxy/auth/auth_checks.py."""
    call_id: Final = new_call_id()
    alias: Final = "alias-" + marker()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias, models=["dec1-embedding"])
        text: Final = marker()
        response: Final = chat(rig, call_id, text, key=key)
        assert response.status_code in (401, 403), response.text
        event: Final = rig.event_mentioning(text, "failure")
        metadata: Final = object_value(event["metadata"])
        assert object_value(event["error_information"])["error_code"] == str(response.status_code)
        assert (metadata["user_api_key_hash"], metadata["user_api_key_alias"]) == (key_hash(key), alias)
        assert all(text not in json.dumps(body) for body in provider_bodies(rig))


def test_unknown_model_is_answered_400_and_makes_no_provider_call(rig: CallbackRig) -> None:
    """A caller asks for a model name no deployment serves. The caller gets a 400 whose message names the model, and
    the provider receives nothing. UNVERIFIED: whether a failure payload is delivered for this case was not traced,
    so the test pins only the response and the absence of a success payload."""
    call_id: Final = new_call_id()
    unknown: Final = "dec1-unknown-" + marker()
    text: Final = marker()
    response: Final = chat(rig, call_id, text, model=unknown)
    assert response.status_code == 400, response.text
    assert unknown in response.text
    assert all(text not in json.dumps(body) for body in provider_bodies(rig))
    assert all(event["status"] != "success" for event in rig.events(call_id))


def test_retry_that_succeeds_logs_one_success_and_one_failure_for_the_first_attempt(rig: CallbackRig) -> None:
    """The caller sends `num_retries: 2` to a deployment whose provider answers 500 once and then 200. The caller
    gets 200 and the provider saw two requests. Exactly one success payload arrives (tokens 11/4, cost 0.019) and
    exactly one failure payload for the failed first attempt, both under the caller's call id.
    UNVERIFIED: read from Logging.should_run_logging (one failure log per logging object) and the @client wrapper
    logging each failed attempt; if the router path now logs differently the failure count is where it shows."""
    call_id: Final = new_call_id()
    text: Final = marker()
    response: Final = chat(rig, call_id, text, model=CHAT_FLAKY, num_retries=2)
    assert response.status_code == 200, response.text
    assert len(tuple(body for body in provider_bodies(rig) if text in json.dumps(body))) == 2
    events: Final = rig.settled_events(call_id, 2)
    assert sorted(str(event["status"]) for event in events) == ["failure", "success"]
    success: Final = next(event for event in events if event["status"] == "success")
    assert (success["prompt_tokens"], success["completion_tokens"]) == (11, 4)
    assert success["response_cost"] == pytest.approx(CHAT_COST)


def test_request_whose_retries_all_fail_logs_one_failure_not_one_per_attempt(rig: CallbackRig) -> None:
    """The caller sends `num_retries: 2` to a deployment whose provider always answers 500. The caller gets 500
    and the provider saw three requests, but the generic_api endpoint must receive exactly one failure payload for
    the call id, not three. UNVERIFIED: derived from the has_logged_async_failure dedupe in
    litellm/litellm_core_utils/litellm_logging.py; a per-attempt logger would deliver three."""
    call_id: Final = new_call_id()
    text: Final = marker()
    response: Final = chat(rig, call_id, text, model=CHAT_OUTAGE, num_retries=2)
    assert response.status_code == 500, response.text
    assert len(tuple(body for body in provider_bodies(rig) if text in json.dumps(body))) == 3
    events: Final = rig.settled_events(call_id, 1)
    assert events[0]["status"] == "failure"


def test_fallback_logs_the_failed_primary_and_the_successful_fallback(rig: CallbackRig) -> None:
    """The requested model group always fails with 500 and is configured to fall back to "dec1-chat". The caller
    gets 200 with `x-litellm-attempted-fallbacks: 1`. Two payloads arrive under the call id: one failure whose
    model_group is the primary group and one success whose model_group is "dec1-chat" with cost 0.019.
    UNVERIFIED: that both payloads share the caller's litellm_call_id was inferred from the proxy passing one
    logging object through the fallback chain."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, model=CHAT_FALLBACK_PRIMARY)
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"
    events: Final = rig.settled_events(call_id, 2)
    by_status: Final = {str(event["status"]): event for event in events}
    assert by_status["failure"]["model_group"] == CHAT_FALLBACK_PRIMARY
    assert by_status["success"]["model_group"] == CHAT
    assert by_status["success"]["response_cost"] == pytest.approx(CHAT_COST)


def test_fallback_payloads_share_one_trace_id(rig: CallbackRig) -> None:
    """The failure payload of the primary and the success payload of the fallback belong to one caller request, so
    both must carry the same non-empty `trace_id`. That is the documented purpose of the field."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, model=CHAT_FALLBACK_PRIMARY).status_code == 200
    events: Final = rig.settled_events(call_id, 2)
    trace_ids: Final = {event["trace_id"] for event in events}
    assert len(trace_ids) == 1 and all(trace_ids), trace_ids


def test_failure_then_success_on_the_same_key_stay_separate(rig: CallbackRig) -> None:
    """One key makes a failing call and then a successful call. Each call id gets its own single payload with its
    own status; the failure does not leak error fields into the next request's success payload."""
    failed, succeeded = new_call_id(), new_call_id()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert chat(rig, failed, model=CHAT_REJECTED, key=key).status_code == 400
        assert chat(rig, succeeded, key=key).status_code == 200
        assert rig.event(failed)["status"] == "failure"
        success: Final = rig.event(succeeded)
        assert (success["status"], success["error_str"]) == ("success", None)
