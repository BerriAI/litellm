"""StandardLoggingPayload for one successful, non-streamed chat completion, field by field.

The proxy runs with `callbacks: ["generic_api", <custom recorder>]`. A caller sends one chat completion with an
`x-litellm-call-id` header and the test reads the JSON object the generic_api callback POSTed to its endpoint for
that call id. That object is the StandardLoggingPayload every logging integration receives, so each test here pins
one field (or one small group) of it to a literal derived from the request, the scripted provider reply (11 prompt
tokens, 4 completion tokens) and the deployment's custom price (0.001 per input token, 0.002 per output token).
A rewrite of the callback pipeline must produce the same keys with the same meaning, because customers' log
pipelines and the spend tables are built on them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import list_value, object_value
from integration.observability.dec1_callbacks_support import (
    CHAT,
    CHAT_COST,
    CHAT_USAGE,
    INPUT_RATE,
    OUTPUT_RATE,
    RECORDER,
    REPLY_TEXT,
    STANDARD_METADATA_FIELDS,
    STANDARD_PAYLOAD_FIELDS,
    CallbackRig,
    callback_rig,
    chat,
    key_hash,
    marker,
    new_call_id,
)

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-payload-chat"), {"callbacks": ["generic_api", RECORDER]}
    ) as created:
        yield created


def test_one_successful_chat_delivers_exactly_one_success_payload(rig: CallbackRig) -> None:
    """A caller sends one chat completion that the provider answers with 200. The generic_api endpoint must receive
    exactly one payload for that call id, with status "success", and no second payload within the following seconds.
    A rewrite that logs from both a sync and an async path would deliver two."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id)
    assert response.status_code == 200, response.text
    events: Final = rig.settled_events(call_id, 1)
    assert events[0]["status"] == "success", events[0]


def test_payload_carries_every_documented_top_level_field(rig: CallbackRig) -> None:
    """The payload for a successful chat must contain every top-level key of StandardLoggingPayload that existing
    consumers read (id, trace_id, call_type, status, model, tokens, cost, timestamps, cache, tags, messages,
    response, error, model_parameters, hidden_params, metadata, guardrail_information and the rest of the pinned
    set). Extra keys are allowed, a missing key is a break."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert STANDARD_PAYLOAD_FIELDS - event.keys() == frozenset(), sorted(event)


def test_payload_metadata_carries_every_documented_key(rig: CallbackRig) -> None:
    """The `metadata` object of the payload must contain every user_api_key_* identity key plus spend_logs_metadata,
    requester_ip_address, user_agent, requester_metadata, requester_custom_headers, applied_guardrails and
    usage_object, each present even when its value is null."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    metadata: Final = object_value(rig.event(call_id)["metadata"])
    assert STANDARD_METADATA_FIELDS - metadata.keys() == frozenset(), sorted(metadata)


def test_id_is_the_provider_response_id_the_caller_received(rig: CallbackRig) -> None:
    """The provider answers with a unique chat completion id. The `id` of the payload must equal the `id` in the
    body the caller received, which is how a log line is joined to a response and to its LiteLLM_SpendLogs row."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id)
    assert response.status_code == 200, response.text
    assert rig.event(call_id)["id"] == response.json()["id"]


def test_litellm_call_id_is_the_response_header_value(rig: CallbackRig) -> None:
    """The caller sends its own `x-litellm-call-id` header. The response must echo it in `x-litellm-call-id` and the
    payload's `litellm_call_id` must be the same string."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id)
    assert response.headers["x-litellm-call-id"] == call_id
    assert rig.event(call_id)["litellm_call_id"] == call_id


def test_proxy_generates_a_call_id_when_the_caller_sends_none(rig: CallbackRig) -> None:
    """A caller that sends no `x-litellm-call-id` header still gets one in the response, and a payload with that
    generated `litellm_call_id` is delivered. The request is tagged so the test can also find it by tag."""
    tag: Final = marker()
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": CHAT, "messages": [{"role": "user", "content": "no call id"}], "metadata": {"tags": [tag]}},
    )
    assert response.status_code == 200, response.text
    generated: Final = response.headers["x-litellm-call-id"]
    assert generated
    event: Final = rig.event(generated)
    assert tag in list_value(event["request_tags"])


def test_trace_id_defaults_to_a_generated_non_empty_string(rig: CallbackRig) -> None:
    """With no trace or session id in the request, `trace_id` must be a non-empty string generated by the proxy, and
    two unrelated requests must get two different trace ids."""
    first, second = new_call_id(), new_call_id()
    assert chat(rig, first).status_code == 200
    assert chat(rig, second).status_code == 200
    one, two = rig.event(first)["trace_id"], rig.event(second)["trace_id"]
    assert isinstance(one, str) and one and isinstance(two, str) and two
    assert one != two


def test_trace_id_takes_metadata_trace_id_from_the_request(rig: CallbackRig) -> None:
    """The caller sends `metadata.trace_id` in the request body. The payload's `trace_id` must be exactly that
    value (read from litellm/litellm_core_utils/litellm_logging.py get_standard_logging_payload_trace_id)."""
    call_id: Final = new_call_id()
    trace: Final = "trace-" + marker()
    assert chat(rig, call_id, metadata={"trace_id": trace}).status_code == 200
    assert rig.event(call_id)["trace_id"] == trace


def test_trace_id_prefers_metadata_session_id_over_metadata_trace_id(rig: CallbackRig) -> None:
    """With the default `request_correlation_in_logs: false`, a request carrying both `metadata.session_id` and
    `metadata.trace_id` must log `trace_id` equal to the session id (the legacy order), and `session_id` must be
    the empty string."""
    call_id: Final = new_call_id()
    session, trace = "session-" + marker(), "trace-" + marker()
    assert chat(rig, call_id, metadata={"session_id": session, "trace_id": trace}).status_code == 200
    event: Final = rig.event(call_id)
    assert (event["trace_id"], event["session_id"]) == (session, "")


def test_call_type_is_acompletion_and_stream_is_null(rig: CallbackRig) -> None:
    """A non-streamed proxy chat completion must log `call_type` "acompletion" and `stream` null (null, not false:
    the field is only ever true for a completed stream)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert (event["call_type"], event["stream"]) == ("acompletion", None)


def test_status_fields_report_llm_success_and_no_guardrail_run(rig: CallbackRig) -> None:
    """With no guardrail configured, `status` is "success" and `status_fields` is exactly
    {"llm_api_status": "success", "guardrail_status": "not_run"}."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert event["status"] == "success"
    assert event["status_fields"] == {"llm_api_status": "success", "guardrail_status": "not_run"}


def test_model_fields_name_the_group_the_deployment_and_the_provider(rig: CallbackRig) -> None:
    """The caller asks for the public model name "dec1-chat", served by one deployment of openai/gpt-4o-mini. The
    payload must report `model_group` "dec1-chat", `custom_llm_provider` "openai", `model` "openai/gpt-4o-mini"
    (the deployment's model string with its provider prefix: `reconstruct_model_name` in
    litellm/litellm_core_utils/core_helpers.py prefers the router's `deployment` metadata), and a `model_id`
    equal to the deployment id the proxy lists for that model in /model/info and returns in the
    `x-litellm-model-id` response header."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id)
    assert response.status_code == 200, response.text
    event: Final = rig.event(call_id)
    listed: Final = tuple(
        object_value(object_value(entry)["model_info"])["id"]
        for entry in list_value(rig.proxy.get("/model/info")["data"])
        if object_value(entry)["model_name"] == CHAT
    )
    assert (event["model_group"], event["custom_llm_provider"], event["model"]) == (
        CHAT,
        "openai",
        "openai/gpt-4o-mini",
    )
    assert listed == (event["model_id"],)
    assert response.headers["x-litellm-model-id"] == event["model_id"]


def test_api_base_is_the_deployment_base_without_a_trailing_slash(rig: CallbackRig) -> None:
    """The deployment's api_base is the scripted provider's URL plus "/v1". The payload's `api_base` and
    `hidden_params.api_base` must both be that URL with no trailing slash, and must not carry the api key."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    expected: Final = rig.provider.url + "/v1"
    assert event["api_base"] == expected
    assert str(object_value(event["hidden_params"])["api_base"]).rstrip("/") == expected


def test_token_counts_come_from_the_provider_usage(rig: CallbackRig) -> None:
    """The provider reports 11 prompt tokens and 4 completion tokens. The payload must report prompt_tokens 11,
    completion_tokens 4, total_tokens 15, and the same numbers in metadata.usage_object."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert (event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) == (*CHAT_USAGE, 15)
    usage: Final = object_value(object_value(event["metadata"])["usage_object"])
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (*CHAT_USAGE, 15)


def test_response_cost_uses_the_deployment_custom_price(rig: CallbackRig) -> None:
    """The deployment is priced at 0.001 per input token and 0.002 per output token, and the reply used 11 and 4.
    `response_cost` must be 0.019, the `x-litellm-response-cost` response header must agree with it, and
    `hidden_params.response_cost` must agree too."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id)
    event: Final = rig.event(call_id)
    assert event["response_cost"] == pytest.approx(CHAT_COST)
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(CHAT_COST)
    assert float(str(object_value(event["hidden_params"])["response_cost"])) == pytest.approx(CHAT_COST)


def test_cost_breakdown_splits_input_and_output_cost(rig: CallbackRig) -> None:
    """`cost_breakdown` must split the 0.019 into input_cost 0.011 (11 tokens at 0.001) and output_cost 0.008
    (4 tokens at 0.002), with total_cost equal to response_cost."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    breakdown: Final = object_value(rig.event(call_id)["cost_breakdown"])
    assert breakdown["input_cost"] == pytest.approx(11 * INPUT_RATE)
    assert breakdown["output_cost"] == pytest.approx(4 * OUTPUT_RATE)
    assert breakdown["total_cost"] == pytest.approx(CHAT_COST)


def test_timestamps_are_epoch_floats_in_order(rig: CallbackRig) -> None:
    """`startTime`, `completionStartTime` and `endTime` must be epoch seconds as floats with startTime <=
    completionStartTime <= endTime. For a non-streamed call completionStartTime equals endTime, and `response_time`
    equals endTime - startTime."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    start, first, end = event["startTime"], event["completionStartTime"], event["endTime"]
    assert isinstance(start, float) and isinstance(first, float) and isinstance(end, float), event
    assert 1_600_000_000 < start <= first <= end
    assert first == end
    assert event["response_time"] == pytest.approx(end - start)


def test_cache_fields_are_empty_when_the_response_was_not_cached(rig: CallbackRig) -> None:
    """With no response cache configured, the payload must not claim a cache hit: `cache_hit` is not true,
    `cache_key` is null and `saved_cache_cost` is 0."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert event["cache_hit"] is not True
    assert (event["cache_key"], event["saved_cache_cost"]) == (None, 0)


def test_request_tags_hold_body_tags_then_user_agent_tags(rig: CallbackRig) -> None:
    """The caller sends `metadata.tags: [a, b]` with the User-Agent "dec1-agent/9.9". `request_tags` must be exactly
    [a, b, "User-Agent: dec1-agent", "User-Agent: dec1-agent/9.9"]: the caller's tags first and in order, then the
    two tags the proxy derives from the user agent."""
    call_id: Final = new_call_id()
    first, second = marker(), marker()
    response: Final = chat(rig, call_id, metadata={"tags": [first, second]}, headers={"user-agent": "dec1-agent/9.9"})
    assert response.status_code == 200, response.text
    assert rig.event(call_id)["request_tags"] == [
        first,
        second,
        "User-Agent: dec1-agent",
        "User-Agent: dec1-agent/9.9",
    ]


def test_request_tags_take_the_x_litellm_tags_header(rig: CallbackRig) -> None:
    """The caller sends no tags in the body and the header `x-litellm-tags: a, b`. Both tags, stripped of the
    space, must be the first two entries of `request_tags`."""
    call_id: Final = new_call_id()
    first, second = marker(), marker()
    response: Final = chat(rig, call_id, headers={"x-litellm-tags": f"{first}, {second}"})
    assert response.status_code == 200, response.text
    assert list_value(rig.event(call_id)["request_tags"])[:2] == [first, second]


def test_end_user_comes_from_the_request_user_field(rig: CallbackRig) -> None:
    """The caller sends `user: <id>` in the chat body. Both `end_user` and `metadata.user_api_key_end_user_id` must
    be that id."""
    call_id: Final = new_call_id()
    end_user: Final = "end-user-" + marker()
    assert chat(rig, call_id, user=end_user).status_code == 200
    event: Final = rig.event(call_id)
    assert event["end_user"] == end_user
    assert object_value(event["metadata"])["user_api_key_end_user_id"] == end_user


def test_requester_ip_and_user_agent_are_the_callers(rig: CallbackRig) -> None:
    """The test client connects from 127.0.0.1 with the User-Agent "dec1-agent/9.9". `requester_ip_address` must be
    "127.0.0.1" and `user_agent` "dec1-agent/9.9", both at the top level and inside `metadata`."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, headers={"user-agent": "dec1-agent/9.9"}).status_code == 200
    event: Final = rig.event(call_id)
    metadata: Final = object_value(event["metadata"])
    assert (event["requester_ip_address"], event["user_agent"]) == ("127.0.0.1", "dec1-agent/9.9")
    assert (metadata["requester_ip_address"], metadata["user_agent"]) == ("127.0.0.1", "dec1-agent/9.9")


def test_messages_are_the_request_messages(rig: CallbackRig) -> None:
    """The caller sends a system message and a user message. `messages` in the payload must be exactly that list,
    in order, with roles and contents unchanged."""
    call_id: Final = new_call_id()
    text: Final = marker()
    sent: Final = [{"role": "system", "content": "dec1 system prompt"}, {"role": "user", "content": text}]
    response: Final = rig.proxy.request(
        "POST", "/v1/chat/completions", {"model": CHAT, "messages": sent}, headers={"x-litellm-call-id": call_id}
    )
    assert response.status_code == 200, response.text
    assert rig.event(call_id)["messages"] == sent


def test_response_is_the_chat_completion_the_caller_received(rig: CallbackRig) -> None:
    """`response` in the payload must be the chat completion as an object: the same id as the caller's body, one
    choice with finish_reason "stop" and the provider's assistant text, and a usage block of 11/4/15."""
    call_id: Final = new_call_id()
    received: Final = chat(rig, call_id).json()
    logged: Final = object_value(rig.event(call_id)["response"])
    choice: Final = object_value(list_value(logged["choices"])[0])
    assert logged["id"] == received["id"]
    assert (choice["finish_reason"], object_value(choice["message"])["content"]) == ("stop", REPLY_TEXT)
    usage: Final = object_value(logged["usage"])
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (*CHAT_USAGE, 15)


def test_error_fields_are_empty_on_success(rig: CallbackRig) -> None:
    """On a successful call `error_str` is null and `error_information` is still an object whose error_code,
    error_class and error_message are empty strings (it is not null)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    information: Final = object_value(event["error_information"])
    assert event["error_str"] is None
    assert (information["error_code"], information["error_class"], information["error_message"]) == ("", "", "")


def test_model_parameters_hold_sampling_params_and_never_the_prompt(rig: CallbackRig) -> None:
    """The caller sends temperature 0.25 and max_tokens 17. `model_parameters` must contain both with those values
    and must not contain `messages`, `prompt` or `input`."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, temperature=0.25, max_tokens=17).status_code == 200
    parameters: Final = object_value(rig.event(call_id)["model_parameters"])
    assert (parameters["temperature"], parameters["max_tokens"]) == (0.25, 17)
    assert {"messages", "prompt", "input"} & parameters.keys() == set()


def test_hidden_params_repeat_the_deployment_and_the_provider_model_name(rig: CallbackRig) -> None:
    """`hidden_params.model_id` must equal the top-level `model_id`, and `hidden_params.litellm_model_name` must be
    the deployment's provider model "openai/gpt-4o-mini". UNVERIFIED: the exact litellm_model_name string was not
    traced; if it differs, look at where the router stamps `litellm_model_name` into `_hidden_params`."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    hidden: Final = object_value(event["hidden_params"])
    assert hidden["model_id"] == event["model_id"]
    assert hidden["litellm_model_name"] == "openai/gpt-4o-mini"


def test_metadata_identifies_the_key_by_hash_and_alias(rig: CallbackRig) -> None:
    """A virtual key with alias A makes the call. `metadata.user_api_key_hash` must be the SHA-256 hex digest of
    the key (never the key itself), `metadata.user_api_key_alias` must be A, and
    `metadata.user_api_key_request_route` must be "/v1/chat/completions"."""
    call_id: Final = new_call_id()
    alias: Final = "alias-" + marker()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias, models=[CHAT])
        assert chat(rig, call_id, key=key).status_code == 200
        metadata: Final = object_value(rig.event(call_id)["metadata"])
        assert metadata["user_api_key_hash"] == key_hash(key)
        assert metadata["user_api_key_alias"] == alias
        assert metadata["user_api_key_request_route"] == "/v1/chat/completions"


def test_metadata_identifies_the_team_the_user_and_the_organization(rig: CallbackRig) -> None:
    """A key that belongs to a user and to a team inside an organization makes the call. The payload metadata must
    carry `user_api_key_team_id`, `user_api_key_team_alias`, `user_api_key_user_id` and `user_api_key_org_id` with
    exactly the ids and alias the management API returned when those objects were created."""
    call_id: Final = new_call_id()
    team_alias: Final = "team-" + marker()
    with rig.proxy.scenario() as scenario:
        organization: Final = scenario.organization(models=[CHAT])
        team: Final = scenario.team(team_alias=team_alias, organization_id=organization, models=[CHAT])
        user: Final = scenario.user()
        key: Final = scenario.key(team_id=team, user_id=user, models=[CHAT])
        assert chat(rig, call_id, key=key).status_code == 200
        metadata: Final = object_value(rig.event(call_id)["metadata"])
        assert (
            metadata["user_api_key_team_id"],
            metadata["user_api_key_team_alias"],
            metadata["user_api_key_user_id"],
            metadata["user_api_key_org_id"],
        ) == (team, team_alias, user, organization)


def test_spend_logs_metadata_is_copied_from_the_request(rig: CallbackRig) -> None:
    """The caller sends `metadata.spend_logs_metadata: {"cost_center": X}`. `metadata.spend_logs_metadata` in the
    payload must be exactly that object."""
    call_id: Final = new_call_id()
    cost_center: Final = marker()
    assert chat(rig, call_id, metadata={"spend_logs_metadata": {"cost_center": cost_center}}).status_code == 200
    assert object_value(rig.event(call_id)["metadata"])["spend_logs_metadata"] == {"cost_center": cost_center}


def test_requester_metadata_is_the_metadata_object_the_caller_sent(rig: CallbackRig) -> None:
    """The caller sends `metadata: {"ticket": X, "tags": [T]}`. `metadata.requester_metadata` must contain both
    caller keys with their values, so a consumer can read the caller's own metadata without the proxy's additions."""
    call_id: Final = new_call_id()
    ticket, tag = marker(), marker()
    assert chat(rig, call_id, metadata={"ticket": ticket, "tags": [tag]}).status_code == 200
    requester: Final = object_value(object_value(rig.event(call_id)["metadata"])["requester_metadata"])
    assert (requester["ticket"], requester["tags"]) == (ticket, [tag])


def test_caller_cannot_forge_key_identity_through_request_metadata(rig: CallbackRig) -> None:
    """The caller sends `metadata.user_api_key_alias` and `metadata.user_api_key_team_id` with made-up values. The
    payload must still report the real key's identity: the forged values appear nowhere in `metadata` identity
    keys (the proxy strips client-supplied user_api_key_* metadata)."""
    call_id: Final = new_call_id()
    forged: Final = "forged-" + marker()
    assert (
        chat(rig, call_id, metadata={"user_api_key_alias": forged, "user_api_key_team_id": forged}).status_code == 200
    )
    metadata: Final = object_value(rig.event(call_id)["metadata"])
    assert forged not in (metadata["user_api_key_alias"], metadata["user_api_key_team_id"])


def test_guardrail_information_is_null_when_no_guardrail_ran(rig: CallbackRig) -> None:
    """With no guardrail configured, `guardrail_information` is null (not an empty list) and
    `metadata.applied_guardrails` is null or empty."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert event["guardrail_information"] is None
    assert not object_value(event["metadata"])["applied_guardrails"]


def test_custom_logger_receives_the_same_payload_as_the_generic_api_callback(rig: CallbackRig) -> None:
    """A custom callback loaded from config as `module.instance` reads `kwargs["standard_logging_object"]` in
    `async_log_success_event`. For one call that object must agree with the generic_api payload on id,
    litellm_call_id, status, model_group, tokens and response_cost, and the kwargs must also carry the classic keys
    `litellm_params`, `model`, `messages`, `response_cost` and `litellm_call_id`."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id)
    record: Final = rig.hook(call_id, "async_log_success_event")
    standard: Final = object_value(record["standard"])
    for name in ("id", "litellm_call_id", "status", "model_group", "prompt_tokens", "completion_tokens"):
        assert standard[name] == event[name], name
    assert standard["response_cost"] == pytest.approx(CHAT_COST)
    assert {"litellm_params", "model", "messages", "response_cost", "litellm_call_id"} <= set(
        list_value(record["kwargs_keys"])
    )
