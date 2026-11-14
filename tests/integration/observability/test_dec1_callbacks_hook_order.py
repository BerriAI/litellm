"""Which CustomLogger hooks fire for a request, how many times, and in what order.

The proxy loads one custom callback from config as `module.instance` (`callbacks: [<recorder>]`, next to
`generic_api`) and one custom pre-call guardrail. The recorder overrides every hook and reports each call with a
sequence number taken inside the single proxy worker, so the order the tests read is the order the proxy made the
calls. Each test sends one request and reads the hook calls recorded for its `x-litellm-call-id`.

The contract, read from litellm/proxy/common_request_processing.py, litellm/proxy/utils.py and
litellm/litellm_core_utils/litellm_logging.py: `async_pre_call_hook` runs first and before any provider call,
`async_moderation_hook` runs alongside the call, `async_pre_call_deployment_hook` and the sync `log_pre_api_call`
run once per provider attempt, `async_post_call_success_hook` runs before the response is returned, the response
headers hook runs for successes and failures, and exactly one of `async_log_success_event` or
`async_log_failure_event` follows. A Rust caller has to drive the same hooks in the same order, because customers'
callbacks depend on seeing a request in pre-call before they see it logged.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration._support.client import held, list_value, object_value
from integration.observability.dec1_callbacks_probe import GUARDRAIL_BLOCK, OUTCOME_HEADER
from integration.observability.dec1_callbacks_support import (
    CHAT,
    CHAT_REJECTED,
    EMBEDDING,
    MESSAGES,
    PROBE_MODULE,
    RECORDER,
    REPLY_TEXT,
    RESPONSES,
    CallbackRig,
    callback_rig,
    chat,
    marker,
    new_call_id,
    post,
    provider_bodies,
    stream_chat,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]

GUARDRAIL_NAME: Final = "dec1-blocking-guardrail"
SUCCESS_HOOKS: Final = (
    "async_pre_call_hook",
    "async_moderation_hook",
    "async_pre_call_deployment_hook",
    "log_pre_api_call",
    "async_post_call_success_deployment_hook",
    "async_post_call_success_hook",
    "async_post_call_response_headers_hook",
    "async_log_success_event",
)
FAILURE_HOOKS: Final = (
    "async_pre_call_hook",
    "async_moderation_hook",
    "async_pre_call_deployment_hook",
    "log_pre_api_call",
    "async_post_call_failure_hook",
    "async_post_call_response_headers_hook",
    "async_log_failure_event",
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    guardrail: Final[dict[str, JsonValue]] = {
        "guardrail_name": GUARDRAIL_NAME,
        "litellm_params": {
            "guardrail": f"{PROBE_MODULE}.BlockingGuardrail",
            "mode": "pre_call",
            "default_on": True,
        },
    }
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-hook-order"),
        {"callbacks": ["generic_api", RECORDER]},
        guardrails=(guardrail,),
    ) as created:
        yield created


def _in_order(names: Sequence[str], *expected: str) -> bool:
    positions: Final = tuple(names.index(name) for name in expected)
    return positions == tuple(sorted(positions))


def test_successful_chat_fires_each_request_hook_exactly_once(rig: CallbackRig) -> None:
    """One successful non-streamed chat. The recorder must see each of these exactly once and nothing else:
    async_pre_call_hook, async_moderation_hook, async_pre_call_deployment_hook, log_pre_api_call,
    async_post_call_success_deployment_hook, async_post_call_success_hook, async_post_call_response_headers_hook,
    async_log_success_event. The set must still be the same three seconds later (no late duplicate)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    rig.settled_hook_names(call_id, SUCCESS_HOOKS)


def test_successful_chat_hooks_run_in_request_order(rig: CallbackRig) -> None:
    """For one successful chat the calls must be ordered: async_pre_call_hook, then
    async_pre_call_deployment_hook, then log_pre_api_call (just before the provider request), then
    async_post_call_success_deployment_hook, then async_post_call_success_hook, then
    async_post_call_response_headers_hook. async_pre_call_hook is the very first hook of the request."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    names: Final = rig.settled_hook_names(call_id, SUCCESS_HOOKS)
    assert names[0] == "async_pre_call_hook", names
    assert _in_order(
        names,
        "async_pre_call_hook",
        "async_pre_call_deployment_hook",
        "log_pre_api_call",
        "async_post_call_success_deployment_hook",
        "async_post_call_success_hook",
        "async_post_call_response_headers_hook",
    ), names


def test_moderation_hook_starts_after_pre_call_and_before_the_post_call_hook(rig: CallbackRig) -> None:
    """async_moderation_hook is started as a task after async_pre_call_hook returned and is awaited before the
    response is processed, so it must be recorded after async_pre_call_hook and before
    async_post_call_success_hook. Its position relative to the deployment hook is not part of the contract."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    names: Final = rig.settled_hook_names(call_id, SUCCESS_HOOKS)
    assert _in_order(names, "async_pre_call_hook", "async_moderation_hook", "async_post_call_success_hook"), names


def test_success_log_event_comes_after_the_provider_call_started(rig: CallbackRig) -> None:
    """async_log_success_event is scheduled in the background once the provider answered, so it must be recorded
    after log_pre_api_call. Where it lands relative to the post-call hooks is not fixed."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    names: Final = rig.settled_hook_names(call_id, SUCCESS_HOOKS)
    assert _in_order(names, "async_pre_call_hook", "log_pre_api_call", "async_log_success_event"), names


def test_async_log_pre_api_call_is_never_dispatched(rig: CallbackRig) -> None:
    """CustomLogger declares `async_log_pre_api_call`, but the logging core only ever calls the sync
    `log_pre_api_call`. A callback that overrides both must receive the sync one once and the async one never.
    This pins today's behaviour so a rewrite does not start double-firing pre-call logs."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    names: Final = rig.settled_hook_names(call_id, SUCCESS_HOOKS)
    assert names.count("log_pre_api_call") == 1
    assert "async_log_pre_api_call" not in names


def test_logger_with_both_sync_and_async_log_hooks_gets_only_the_async_one(rig: CallbackRig) -> None:
    """The recorder overrides both `log_success_event` and `async_log_success_event`. On the proxy (an async
    caller) only the async one may fire for a successful chat, otherwise every dual-implemented logger would log
    each request twice."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    names: Final = rig.settled_hook_names(call_id, SUCCESS_HOOKS)
    assert "log_success_event" not in names and "log_failure_event" not in names


def test_pre_call_hook_receives_the_proxy_call_type_and_the_key_identity(rig: CallbackRig) -> None:
    """async_pre_call_hook for a chat completion must be called with `call_type` "acompletion", a request `data`
    dict that already carries `litellm_call_id` equal to the caller's header and the `litellm_logging_obj`, and a
    `user_api_key_dict` whose key_alias is the calling key's alias."""
    call_id: Final = new_call_id()
    alias: Final = "alias-" + marker()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias)
        assert chat(rig, call_id, key=key).status_code == 200
        record: Final = rig.hook(call_id, "async_pre_call_hook")
        assert (record["hook_call_type"], record["key_alias"], record["has_logging_obj"]) == (
            "acompletion",
            alias,
            True,
        )


@pytest.mark.parametrize(
    ("route", "body", "call_type"),
    [
        pytest.param("/v1/embeddings", {"model": EMBEDDING, "input": "dec1 hook order"}, "aembedding", id="embeddings"),
        pytest.param(
            "/v1/messages",
            {"model": MESSAGES, "max_tokens": 16, "messages": [{"role": "user", "content": "dec1 hook order"}]},
            "anthropic_messages",
            id="messages",
        ),
        pytest.param("/v1/responses", {"model": RESPONSES, "input": "dec1 hook order"}, "aresponses", id="responses"),
    ],
)
def test_other_surfaces_run_pre_call_then_exactly_one_success_log(
    rig: CallbackRig, route: str, body: dict[str, JsonValue], call_type: str
) -> None:
    """One successful embeddings, /v1/messages or /v1/responses request. async_pre_call_hook must fire once with
    that surface's call_type, async_log_success_event must fire exactly once with the same call_type, it must come
    after the pre-call hook, and no failure hook may fire."""
    call_id: Final = new_call_id()
    assert post(rig, route, body, call_id).status_code == 200
    logged: Final = rig.hook(call_id, "async_log_success_event")
    names: Final = rig.hook_names(call_id)
    assert rig.hook(call_id, "async_pre_call_hook")["hook_call_type"] == call_type
    assert logged["call_type"] == call_type
    assert names.count("async_pre_call_hook") == 1 and names.count("async_log_success_event") == 1, names
    assert _in_order(names, "async_pre_call_hook", "async_log_success_event"), names
    assert "async_post_call_failure_hook" not in names and "async_log_failure_event" not in names


def test_provider_failure_fires_the_failure_hooks_and_no_success_hook(rig: CallbackRig) -> None:
    """The provider answers 400. The recorder must see exactly: async_pre_call_hook, async_moderation_hook,
    async_pre_call_deployment_hook, log_pre_api_call, async_post_call_failure_hook,
    async_post_call_response_headers_hook and async_log_failure_event, each once. Neither
    async_post_call_success_hook nor async_log_success_event may fire."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, model=CHAT_REJECTED).status_code == 400
    rig.settled_hook_names(call_id, FAILURE_HOOKS)


def test_provider_failure_runs_the_failure_hook_before_the_response_headers_hook(rig: CallbackRig) -> None:
    """On a provider failure the order is async_pre_call_hook, async_pre_call_deployment_hook, log_pre_api_call,
    async_post_call_failure_hook, async_post_call_response_headers_hook: the failure hook can still transform the
    error before headers are computed for it."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, model=CHAT_REJECTED).status_code == 400
    names: Final = rig.settled_hook_names(call_id, FAILURE_HOOKS)
    assert _in_order(
        names,
        "async_pre_call_hook",
        "async_pre_call_deployment_hook",
        "log_pre_api_call",
        "async_post_call_failure_hook",
        "async_post_call_response_headers_hook",
    ), names


def test_failure_hook_receives_the_exception_class_and_status(rig: CallbackRig) -> None:
    """For a provider 400, async_post_call_failure_hook must be handed the litellm BadRequestError with
    status_code 400, and a request_data dict that no longer contains the non-serialisable `litellm_logging_obj`."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, model=CHAT_REJECTED).status_code == 400
    record: Final = rig.hook(call_id, "async_post_call_failure_hook")
    assert (record["exception_class"], record["exception_status"], record["has_logging_obj"]) == (
        "BadRequestError",
        400,
        False,
    )


def test_response_headers_hook_headers_reach_the_caller_on_success_and_failure(rig: CallbackRig) -> None:
    """The recorder's async_post_call_response_headers_hook returns one header whose value is "success" when it
    was handed a response and "failure" when it was handed None. A 200 chat response must carry that header with
    "success" and a provider-400 error response must carry it with "failure"."""
    ok: Final = chat(rig, new_call_id())
    failed: Final = chat(rig, new_call_id(), model=CHAT_REJECTED)
    assert (ok.status_code, ok.headers[OUTCOME_HEADER]) == (200, "success")
    assert (failed.status_code, failed.headers[OUTCOME_HEADER]) == (400, "failure")


def test_response_headers_hook_receives_routing_info_on_success(rig: CallbackRig) -> None:
    """On success the response headers hook gets `litellm_call_info` naming the provider and the deployment that
    served the call: custom_llm_provider "openai", api_base equal to the deployment base, and a model_id equal to
    the `x-litellm-model-id` response header. It is also handed the caller's request headers."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id)
    record: Final = rig.hook(call_id, "async_post_call_response_headers_hook")
    info: Final = object_value(record["call_info"])
    assert info["custom_llm_provider"] == "openai"
    assert str(info["api_base"]).rstrip("/") == rig.provider.url + "/v1"
    assert info["model_id"] == response.headers["x-litellm-model-id"]
    assert "x-litellm-call-id" in list_value(record["saw_request_headers"])


def test_streamed_chat_fires_one_success_log_after_the_stream_and_no_stream_event(rig: CallbackRig) -> None:
    """A chat completion is streamed to the end. The recorder must get async_pre_call_hook,
    async_pre_call_deployment_hook, log_pre_api_call, async_post_call_streaming_iterator_hook and the response
    headers hook once each, in that order, then exactly one async_log_success_event with the assembled response,
    and still exactly one three seconds later. It must get no async_log_stream_event or log_stream_event (the
    proxy logs a stream once, not per chunk) and no async_post_call_success_hook (the streaming branch returns
    before that hook; it is for non-streamed responses)."""
    call_id: Final = new_call_id()
    assert stream_chat(rig, call_id)[0] == 200
    once: Final = (
        "async_pre_call_hook",
        "async_pre_call_deployment_hook",
        "log_pre_api_call",
        "async_post_call_response_headers_hook",
        "async_post_call_streaming_iterator_hook",
        "async_log_success_event",
    )
    names: Final = held(
        lambda: rig.hook_names(call_id),
        lambda found: all(found.count(name) == 1 for name in once),
        holding=3,
        seconds=30,
    )
    assert _in_order(names, *once), names
    for absent in ("async_log_stream_event", "log_stream_event", "async_post_call_success_hook"):
        assert absent not in names, names


def test_streamed_chat_headers_hook_runs_before_the_first_chunk_is_sent(rig: CallbackRig) -> None:
    """For a stream the response headers hook must run before the body starts, so its header is on the streaming
    response the caller receives, with the value "success"."""
    status, headers, _ = stream_chat(rig, new_call_id())
    assert (status, headers[OUTCOME_HEADER]) == (200, "success")


def test_streaming_chunk_hook_sees_the_text_so_far_for_every_chunk(rig: CallbackRig) -> None:
    """The provider streams the reply as two content chunks followed by a finish chunk. For a stream made with a
    uniquely aliased key, async_post_call_streaming_hook must be called once per chunk with the text accumulated
    so far including that chunk: every recorded text is a prefix of the next one, the first is the first half of
    the reply and the last is the whole reply."""
    alias: Final = "alias-" + marker()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias)
        call_id: Final = new_call_id()
        assert stream_chat(rig, call_id, key=key)[0] == 200
        rig.hook(call_id, "async_log_success_event")
        texts: Final = rig.stream_hook_texts(alias)
        assert len(texts) >= 3, texts
        assert all(later.startswith(earlier) for earlier, later in itertools.pairwise(texts)), texts
        assert (texts[0], texts[-1]) == (REPLY_TEXT[: len(REPLY_TEXT) // 2], REPLY_TEXT)


def test_guardrail_blocked_request_never_reaches_the_provider_hooks(rig: CallbackRig) -> None:
    """A pre-call guardrail rejects the request with 400. The caller gets 400 with the guardrail's message, the
    provider receives nothing, and the recorder sees no async_pre_call_deployment_hook, no
    async_post_call_success_hook and no async_log_success_event for the call.

    Pinned current behaviour: log_pre_api_call does fire, exactly once and after async_pre_call_hook. For a
    request stopped by the proxy, `_handle_logging_proxy_only_error` (litellm/proxy/utils.py) calls
    `Logging.pre_call` on purpose so every callback's input hook sees the request before its failure is logged,
    even though no provider attempt is made. A caller that counts log_pre_api_call as provider attempts sees one
    here."""
    call_id: Final = new_call_id()
    text: Final = f"{GUARDRAIL_BLOCK} {marker()}"
    response: Final = chat(rig, call_id, text)
    assert response.status_code == 400, response.text
    assert "blocked by the dec1 guardrail" in response.text
    rig.hook(call_id, "async_log_failure_event")
    names: Final = rig.hook_names(call_id)
    assert all(text not in str(body) for body in provider_bodies(rig))
    for absent in (
        "async_pre_call_deployment_hook",
        "async_post_call_success_hook",
        "async_log_success_event",
    ):
        assert absent not in names, names
    assert names.count("log_pre_api_call") == 1, names
    assert names.index("async_pre_call_hook") < names.index("log_pre_api_call"), names


def test_guardrail_blocked_request_fires_failure_hook_and_one_failure_log(rig: CallbackRig) -> None:
    """For the guardrail-blocked request the recorder must get async_post_call_failure_hook exactly once (with an
    HTTPException of status 400) and async_log_failure_event exactly once, and the response headers hook must mark
    the response "failure"."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, f"{GUARDRAIL_BLOCK} {marker()}")
    assert (response.status_code, response.headers[OUTCOME_HEADER]) == (400, "failure")
    rig.settled_events(call_id, 1)
    names: Final = rig.hook_names(call_id)
    assert names.count("async_post_call_failure_hook") == 1 and names.count("async_log_failure_event") == 1, names
    assert rig.hook(call_id, "async_post_call_failure_hook")["exception_status"] == 400


def test_guardrail_blocked_payload_records_the_guardrail_intervention(rig: CallbackRig) -> None:
    """The failure payload for a guardrail-blocked request must name the guardrail: `guardrail_information` is a
    list with one entry whose guardrail_name is the configured name, guardrail_mode is "pre_call" and
    guardrail_status is "guardrail_intervened"; `status_fields.guardrail_status` is "guardrail_intervened" and
    `status` is "failure"."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, f"{GUARDRAIL_BLOCK} {marker()}").status_code == 400
    event: Final = rig.event(call_id, "failure")
    entries: Final = list_value(event["guardrail_information"])
    assert len(entries) == 1, entries
    entry: Final = object_value(entries[0])
    assert (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"]) == (
        GUARDRAIL_NAME,
        "pre_call",
        "guardrail_intervened",
    )
    assert object_value(event["status_fields"])["guardrail_status"] == "guardrail_intervened"


def test_guardrail_that_passes_is_recorded_as_success_on_the_success_payload(rig: CallbackRig) -> None:
    """The same guardrail lets an ordinary request through. The success payload must carry one
    `guardrail_information` entry with the guardrail's name and guardrail_status "success",
    `status_fields.guardrail_status` "success", and the guardrail's name in `metadata.applied_guardrails`."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    event: Final = rig.event(call_id, "success")
    entry: Final = object_value(list_value(event["guardrail_information"])[0])
    assert (entry["guardrail_name"], entry["guardrail_status"]) == (GUARDRAIL_NAME, "success")
    assert object_value(event["status_fields"])["guardrail_status"] == "success"
    assert GUARDRAIL_NAME in list_value(object_value(event["metadata"])["applied_guardrails"])


def test_request_rejected_by_key_auth_skips_pre_call_and_logs_one_failure(rig: CallbackRig) -> None:
    """A request with an unknown virtual key is rejected with 401 before the request pipeline starts. The recorder
    must get no async_pre_call_hook and no async_pre_call_deployment_hook for it, but must get exactly one
    async_log_failure_event, so auth failures stay visible to logging callbacks. Key auth fails before the endpoint
    reads `x-litellm-call-id`, so the hooks are found by the request's unique text, not by the call id.

    Pinned current behaviour: log_pre_api_call fires once even though no provider attempt is made, because
    `_handle_logging_proxy_only_error` (litellm/proxy/utils.py) calls `Logging.pre_call` before logging a proxy
    rejection."""
    text: Final = marker()
    assert chat(rig, new_call_id(), text, key="sk-dec1-unknown-" + marker()).status_code == 401
    names: Final = held(
        lambda: rig.hook_names_mentioning(text),
        lambda found: found.count("async_log_failure_event") == 1,
        holding=3,
        seconds=30,
    )
    for absent in ("async_pre_call_hook", "async_pre_call_deployment_hook"):
        assert absent not in names, names
    assert names.count("log_pre_api_call") == 1, names


def test_two_instances_in_callbacks_each_get_every_hook_once(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Two custom callback instances are listed in `callbacks`. For one successful chat each instance must get its
    own async_pre_call_hook and its own async_log_success_event exactly once, and the first listed instance's
    pre-call hook must run before the second's."""
    second: Final = f"{PROBE_MODULE}.second_recorder"
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-two-instances"), {"callbacks": [RECORDER, second]}
    ) as two:
        call_id: Final = new_call_id()
        assert chat(two, call_id, model=CHAT).status_code == 200
        first_pre: Final = two.hook(call_id, "async_pre_call_hook", "recorder")
        second_pre: Final = two.hook(call_id, "async_pre_call_hook", "second_recorder")
        two.hook(call_id, "async_log_success_event", "recorder")
        two.hook(call_id, "async_log_success_event", "second_recorder")
        assert int(str(first_pre["seq"])) < int(str(second_pre["seq"]))
        for logger in ("recorder", "second_recorder"):
            names = two.hook_names(call_id, logger)
            assert names.count("async_pre_call_hook") == 1 and names.count("async_log_success_event") == 1, names
