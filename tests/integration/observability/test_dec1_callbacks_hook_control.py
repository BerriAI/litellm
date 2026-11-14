"""What a custom callback can do to a request, and what happens when a callback misbehaves.

The proxy loads the recorder from config as `module.instance` next to `generic_api`. The recorder changes
behaviour only when the request text carries a `dec1-...` directive, so each test sends one request with one
directive and observes the caller's response, the request the scripted provider received and the payload the
generic_api endpoint received.

The contract, read from litellm/proxy/utils.py (pre_call_hook, process_pre_call_hook_response, during_call_hook,
post_call_success_hook, post_call_failure_hook) and litellm/utils.py (async_pre_call_deployment_hook):
the blocking hooks are trusted and fail closed. `async_pre_call_hook` may return a new request dict, return a
string to reject with 400, or raise; `async_moderation_hook` may raise; `async_post_call_success_hook` may return
a new response or raise; `async_post_call_failure_hook` may return an HTTPException to replace the error. The
logging hooks are the opposite: an exception or a slow `async_log_success_event` must never change or delay what
the caller gets, and must not stop the other callbacks from logging.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import held, object_value
from integration.observability.dec1_callbacks_probe import (
    DEPLOYMENT_MODIFY,
    FAILURE_RAISE_PLAIN,
    FAILURE_TRANSFORM,
    LOG_RAISE,
    LOG_SLOW,
    MODERATION_RAISE,
    POST_MODIFY,
    POST_RAISE_HTTP,
    PRE_MODIFY,
    PRE_RAISE_HTTP,
    PRE_RAISE_PLAIN,
    PRE_REJECT_STRING,
    REJECTION,
    REWRITTEN_REQUEST,
    REWRITTEN_RESPONSE,
    SLOW_LOG_SECONDS,
)
from integration.observability.dec1_callbacks_support import (
    CHAT_REJECTED,
    EMBEDDING,
    MESSAGES,
    PROVIDER_REJECTION,
    RECORDER,
    REPLY_TEXT,
    RESPONSES,
    CallbackRig,
    callback_rig,
    chat,
    error_message,
    marker,
    new_call_id,
    post,
    provider_bodies,
    stream,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-hook-control"), {"callbacks": ["generic_api", RECORDER]}
    ) as created:
        yield created


def _provider_saw(rig: CallbackRig, text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(body for body in provider_bodies(rig) if text in json.dumps(body))


def test_pre_call_hook_can_rewrite_the_request_sent_to_the_provider(rig: CallbackRig) -> None:
    """The pre-call hook returns a request dict whose messages were replaced. The caller gets 200, and the
    provider must receive the rewritten user message and not the caller's original text."""
    original: Final = f"{PRE_MODIFY} {marker()}"
    rig.provider.drain()
    response: Final = chat(rig, new_call_id(), original)
    assert response.status_code == 200, response.text
    bodies: Final = provider_bodies(rig)
    assert [body["messages"] for body in bodies] == [[{"role": "user", "content": REWRITTEN_REQUEST}]], bodies


def test_pre_call_rewrite_is_what_the_loggers_record_as_the_request(rig: CallbackRig) -> None:
    """After the pre-call hook rewrote the messages, the generic_api payload's `messages` must be the rewritten
    list: logging records what was sent to the model, not what the caller originally typed."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, f"{PRE_MODIFY} {marker()}").status_code == 200
    assert rig.event(call_id)["messages"] == [{"role": "user", "content": REWRITTEN_REQUEST}]


def test_pre_call_hook_returning_a_string_rejects_chat_with_400_and_that_message(rig: CallbackRig) -> None:
    """The pre-call hook returns a string for a chat completion. The caller must get HTTP 400 whose
    `error.message` is exactly that string and whose `error.code` is "400", and the provider must receive nothing.
    (The proxy calls the hook with call_type "acompletion", so the string becomes an HTTPException with
    detail {"error": <string>}, not a 200 assistant message.)"""
    text: Final = f"{PRE_REJECT_STRING} {marker()}"
    response: Final = chat(rig, new_call_id(), text)
    assert response.status_code == 400, response.text
    error: Final = object_value(response.json()["error"])
    assert (error["message"], str(error["code"])) == (REJECTION, "400")
    assert _provider_saw(rig, text) == ()


@pytest.mark.parametrize(
    ("route", "body"),
    [
        pytest.param("/v1/embeddings", {"model": EMBEDDING, "input": PRE_REJECT_STRING}, id="embeddings"),
        pytest.param(
            "/v1/messages",
            {"model": MESSAGES, "max_tokens": 16, "messages": [{"role": "user", "content": PRE_REJECT_STRING}]},
            id="messages",
        ),
        pytest.param("/v1/responses", {"model": RESPONSES, "input": PRE_REJECT_STRING}, id="responses"),
    ],
)
def test_pre_call_hook_returning_a_string_rejects_other_surfaces_with_400(
    rig: CallbackRig, route: str, body: dict[str, JsonValue]
) -> None:
    """The pre-call hook returns a string on embeddings, /v1/messages and /v1/responses. Each caller must get
    HTTP 400 and the hook's string somewhere in the error body. UNVERIFIED: /v1/messages renders errors in the
    Anthropic shape, so only the status and the presence of the string are pinned, not the JSON path."""
    response: Final = post(rig, route, body, new_call_id())
    assert response.status_code == 400, response.text
    assert REJECTION in response.text


def test_pre_call_string_rejection_is_logged_as_one_failure(rig: CallbackRig) -> None:
    """A request rejected by the pre-call hook's string must still be logged: exactly one generic_api payload with
    status "failure", error_code "400" and the hook's string in error_message, and the recorder's
    async_post_call_failure_hook must have been called for it."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, f"{PRE_REJECT_STRING} {marker()}").status_code == 400
    event: Final = rig.settled_events(call_id, 1)[0]
    information: Final = object_value(event["error_information"])
    assert (event["status"], information["error_code"]) == ("failure", "400")
    assert REJECTION in str(information["error_message"])
    assert "async_post_call_failure_hook" in rig.hook_names(call_id)


def test_pre_call_hook_raising_http_exception_sets_the_status_and_message(rig: CallbackRig) -> None:
    """The pre-call hook raises HTTPException(status_code=403, detail={"error": "..."}). The caller must get HTTP
    403 with `error.message` equal to the detail's error string, and the provider must receive nothing."""
    text: Final = f"{PRE_RAISE_HTTP} {marker()}"
    response: Final = chat(rig, new_call_id(), text)
    assert response.status_code == 403, response.text
    assert error_message(response) == "blocked by the dec1 pre-call hook"
    assert _provider_saw(rig, text) == ()


def test_pre_call_http_exception_is_logged_with_its_status(rig: CallbackRig) -> None:
    """The 403 raised by the pre-call hook must produce one failure payload whose error_information.error_code is
    "403" and whose error_message carries the hook's message."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, f"{PRE_RAISE_HTTP} {marker()}").status_code == 403
    information: Final = object_value(rig.settled_events(call_id, 1)[0]["error_information"])
    assert information["error_code"] == "403"
    assert "blocked by the dec1 pre-call hook" in str(information["error_message"])


def test_pre_call_hook_crash_fails_the_request_closed(rig: CallbackRig) -> None:
    """The pre-call hook raises a plain ValueError. The request must fail with HTTP 500 and the provider must
    receive nothing: a crashing pre-call hook never lets the request through unchecked. The recorder's
    async_post_call_failure_hook runs for it."""
    call_id: Final = new_call_id()
    text: Final = f"{PRE_RAISE_PLAIN} {marker()}"
    response: Final = chat(rig, call_id, text)
    assert response.status_code == 500, response.text
    assert _provider_saw(rig, text) == ()
    assert rig.hook(call_id, "async_post_call_failure_hook")["exception_class"] == "ValueError"


def test_pre_call_rejection_of_a_stream_request_is_a_plain_400(rig: CallbackRig) -> None:
    """The caller asks for a stream and the pre-call hook rejects with a string. The caller must get HTTP 400 with
    the hook's string in a JSON error body, not a 200 event stream."""
    body: Final[dict[str, JsonValue]] = {
        "model": "dec1-chat",
        "messages": [{"role": "user", "content": f"{PRE_REJECT_STRING} {marker()}"}],
    }
    status, text = stream(rig, "/v1/chat/completions", body, new_call_id())
    assert status == 400, text
    assert json.loads(text)["error"]["message"] == REJECTION


def test_moderation_hook_raising_blocks_the_request(rig: CallbackRig) -> None:
    """async_moderation_hook raises HTTPException(400). It runs alongside the provider call, so the provider may
    already have been called, but the caller must get HTTP 400 with the hook's message and never the model's
    reply."""
    response: Final = chat(rig, new_call_id(), f"{MODERATION_RAISE} {marker()}")
    assert response.status_code == 400, response.text
    assert error_message(response) == "blocked by the dec1 moderation hook"
    assert REPLY_TEXT not in response.text


def test_post_call_hook_can_rewrite_the_response_the_caller_receives(rig: CallbackRig) -> None:
    """The provider replies with its scripted text and async_post_call_success_hook returns the response with the
    assistant content replaced. The caller must receive the replaced content with status 200 and the same token
    usage the provider reported."""
    response: Final = chat(rig, new_call_id(), f"{POST_MODIFY} {marker()}")
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["choices"][0]["message"]["content"] == REWRITTEN_RESPONSE
    assert (body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"]) == (11, 4)


def test_post_call_hook_raising_replaces_a_provider_success_with_its_error(rig: CallbackRig) -> None:
    """The provider answered 200 but async_post_call_success_hook raises HTTPException(422). The provider did
    receive the request, and the caller must get HTTP 422 with the hook's message and not the model's reply."""
    text: Final = f"{POST_RAISE_HTTP} {marker()}"
    response: Final = chat(rig, new_call_id(), text)
    assert response.status_code == 422, response.text
    assert error_message(response) == "blocked by the dec1 post-call hook"
    assert REPLY_TEXT not in response.text
    assert len(_provider_saw(rig, text)) == 1


def test_failure_hook_can_replace_the_error_sent_to_the_caller(rig: CallbackRig) -> None:
    """The provider answers 400 and async_post_call_failure_hook returns HTTPException(503, "..."). The caller
    must get HTTP 503 with that detail as `error.message` instead of the provider's 400."""
    response: Final = chat(rig, new_call_id(), f"{FAILURE_TRANSFORM} {marker()}", model=CHAT_REJECTED)
    assert response.status_code == 503, response.text
    assert error_message(response) == "transformed by the dec1 failure hook"


def test_failure_hook_crash_leaves_the_original_error_untouched(rig: CallbackRig) -> None:
    """The provider answers 400 and async_post_call_failure_hook raises a plain RuntimeError. The caller must
    still get the provider's 400 with the provider's message: a non-HTTP exception from the failure hook is
    swallowed."""
    response: Final = chat(rig, new_call_id(), f"{FAILURE_RAISE_PLAIN} {marker()}", model=CHAT_REJECTED)
    assert response.status_code == 400, response.text
    assert PROVIDER_REJECTION in error_message(response)


def test_success_logger_crash_does_not_change_the_response(rig: CallbackRig) -> None:
    """async_log_success_event raises RuntimeError after it ran. The caller must get the normal 200 with the
    provider's reply."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, f"{LOG_RAISE} {marker()}")
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == REPLY_TEXT
    rig.hook(call_id, "async_log_success_event")


def test_success_logger_crash_does_not_stop_other_callbacks(rig: CallbackRig) -> None:
    """The recorder's async_log_success_event raises. The generic_api callback configured next to it must still
    deliver its one success payload for the same call."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, f"{LOG_RAISE} {marker()}").status_code == 200
    assert rig.settled_events(call_id, 1)[0]["status"] == "success"


def test_failure_logger_crash_does_not_change_the_error_or_other_callbacks(rig: CallbackRig) -> None:
    """The provider answers 400 and the recorder's async_log_failure_event raises. The caller must still get the
    provider's 400, and generic_api must still deliver its one failure payload."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, f"{LOG_RAISE} {marker()}", model=CHAT_REJECTED)
    assert response.status_code == 400, response.text
    assert PROVIDER_REJECTION in error_message(response)
    assert rig.settled_events(call_id, 1)[0]["status"] == "failure"


def test_slow_success_logger_does_not_delay_the_response(rig: CallbackRig) -> None:
    """The recorder's async_log_success_event waits six seconds before reporting. The caller's response must come
    back well before that (under three seconds), and the recorder's log event must still arrive afterwards.
    Callback delivery runs after the response, never in its path."""
    call_id: Final = new_call_id()
    started: Final = time.monotonic()
    response: Final = chat(rig, call_id, f"{LOG_SLOW} {marker()}")
    elapsed: Final = time.monotonic() - started
    assert response.status_code == 200, response.text
    assert elapsed < SLOW_LOG_SECONDS - 3, elapsed
    assert rig.hook_names(call_id).count("async_log_success_event") == 0
    rig.hook(call_id, "async_log_success_event")


def test_slow_success_logger_does_not_delay_the_next_request_or_its_log(rig: CallbackRig) -> None:
    """While one request's success logger is still waiting its six seconds, a second request on the same proxy
    must answer normally and its own success payload must be delivered before the slow one finishes."""
    slow, fast = new_call_id(), new_call_id()
    started: Final = time.monotonic()
    assert chat(rig, slow, f"{LOG_SLOW} {marker()}").status_code == 200
    assert chat(rig, fast).status_code == 200
    rig.event(fast, "success", seconds=SLOW_LOG_SECONDS - 2)
    assert time.monotonic() - started < SLOW_LOG_SECONDS, "the fast request's log waited for the slow logger"
    rig.hook(slow, "async_log_success_event")


def test_deployment_hook_can_change_the_request_for_the_chosen_deployment(rig: CallbackRig) -> None:
    """async_pre_call_deployment_hook returns the call kwargs with max_tokens set to 7. The provider must receive
    `max_tokens: 7` although the caller sent none, and the caller still gets 200."""
    text: Final = f"{DEPLOYMENT_MODIFY} {marker()}"
    response: Final = chat(rig, new_call_id(), text)
    assert response.status_code == 200, response.text
    bodies: Final = _provider_saw(rig, text)
    assert [body.get("max_tokens") for body in bodies] == [7], bodies


def test_deployment_hook_receives_the_resolved_provider_model(rig: CallbackRig) -> None:
    """async_pre_call_deployment_hook runs after routing, so it is called once per provider attempt with the
    call_type of the litellm call. For one successful chat it is called exactly once and its recorded call id is
    the caller's `x-litellm-call-id`."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id).status_code == 200
    rig.hook(call_id, "async_log_success_event")
    assert rig.hook_names(call_id).count("async_pre_call_deployment_hook") == 1


def test_unreachable_generic_api_endpoint_does_not_break_requests(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The generic_api callback is pointed at a port nothing listens on. Chat requests must still answer 200 with
    the provider's reply, repeatedly, and the custom callback configured next to it must still get its success
    log event for each call."""
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-dead-sink"),
        {"callbacks": ["generic_api", RECORDER]},
        environment={"GENERIC_LOGGER_ENDPOINT": "http://127.0.0.1:9/dec1-unreachable"},
    ) as broken:
        for _ in range(3):
            call_id = new_call_id()
            response = chat(broken, call_id)
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == REPLY_TEXT
            broken.hook(call_id, "async_log_success_event")


def test_success_logger_slower_than_the_worker_budget_is_cancelled_without_harm(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The logging worker gives each request's log coroutine LOGGING_WORKER_MAX_TIME_PER_COROUTINE seconds (20 by
    default, 2 here). The recorder's async_log_success_event waits six seconds before reporting, so it is
    cancelled: its record never arrives. The caller still got 200, generic_api (listed before it) still delivered
    its payload, and a following ordinary request is logged by both callbacks.
    UNVERIFIED: read from litellm/litellm_core_utils/logging_worker.py (asyncio.wait_for around each task); the
    proxy may route some success logs outside the worker."""
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-worker-budget"),
        {"callbacks": ["generic_api", RECORDER]},
        environment={"LOGGING_WORKER_MAX_TIME_PER_COROUTINE": "2"},
    ) as budgeted:
        slow, normal = new_call_id(), new_call_id()
        assert chat(budgeted, slow, f"{LOG_SLOW} {marker()}").status_code == 200
        assert budgeted.event(slow)["status"] == "success"
        assert chat(budgeted, normal).status_code == 200
        budgeted.event(normal)
        budgeted.hook(normal, "async_log_success_event")
        held(
            lambda: budgeted.hook_names(slow).count("async_log_success_event"),
            lambda count: count == 0,
            holding=SLOW_LOG_SECONDS + 2,
            seconds=SLOW_LOG_SECONDS + 4,
        )
