"""CustomLoggerV2 callbacks configured in `litellm_settings.callbacks` as `module.instance` must behave on the proxy
the way the user guide (litellm/integrations/custom_logger_v2/user-guide.md, contract PRs #51214 and #51526)
describes, and legacy loggers in the same list must keep their position.

Contract under test:
- a `pre_call` handler declared `may_abort=True` that returns `Abort(status, message, headers)` stops the request
  with that status, `error.message` and headers, nothing after it on that hook runs, and the provider gets
  nothing; `Abort(exception=...)` raises that exception instead
- a `pre_call` handler that writes `messages` replaces the request the provider receives, and a legacy logger
  listed after it sees the rewrite
- an exception raised by a `pre_call` handler fails the request (request-path hooks propagate)
- a `response_headers` patch reaches the caller; a `post_call_success` patch replaces the reply
- an `attempt_ended` reader is called once per attempt with outcome, usage and cost, off the request path; its
  crash never fails the request; `honors_no_log` defaults to False and only an explicit True skips `no-log` calls

The V2 loggers live in `dec1_cblv2_probe` and report to the probe sink with the request text, because the guide
says `ctx.call` may be None on proxy hooks. The guide marks config registration of a V2 instance as "not run in a
live proxy for this guide"; these tests are that run.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import eventually, held
from integration.observability.dec1_cblv2_probe import (
    V2_ABORT,
    V2_ABORT_CUSTOM_EXCEPTION,
    V2_ABORT_HEADER,
    V2_ABORT_MESSAGE,
    V2_ABORT_STATUS,
    V2_EXCEPTION_MESSAGE,
    V2_EXCEPTION_STATUS,
    V2_METADATA_PATCH,
    V2_PATCH,
    V2_POST_PATCH,
    V2_RAISE,
    V2_RESPONSE_HEADER,
    V2_REWRITTEN_REQUEST,
    V2_REWRITTEN_RESPONSE,
)
from integration.observability.dec1_cblv2_support import (
    CHAT,
    CHAT_COST,
    CHAT_FALLBACK_PRIMARY,
    CHAT_REJECTED,
    PROBE_MODULE,
    REPLY_TEXT,
    Rig,
    cblv2_rig,
    chat,
    error_message,
    marker,
    reply_text,
    skip_until_v2_instances_load,
    stream_chat,
)
from pydantic import JsonValue

LICENSED: Final = bool(os.environ.get("LITELLM_LICENSE"))
LOGGERS: Final = (
    "v2_gate",
    "chain_first",
    "v2_post_gate",
    "v2_reader",
    "v2_no_log_reader",
    "v2_crashing_reader",
    "v2_awaited_reader",
    "v2_metadata_gate",
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    skip_until_v2_instances_load()
    directory: Final = tmp_path_factory.mktemp("cblv2-v2-loggers")
    with cblv2_rig(directory, {"callbacks": [f"{PROBE_MODULE}.{name}" for name in LOGGERS]}) as booted:
        yield booted


def _attempts(rig: Rig, text: str, logger: str, count: int) -> tuple[dict[str, JsonValue], ...]:
    """The `attempt_ended` records one reader sent for the request carrying `text`, once exactly `count` arrived
    and no more arrive for 3 seconds."""
    return held(
        lambda: tuple(record for record in rig.mentioning(text, logger) if record["hook"] == "attempt_ended"),
        lambda records: len(records) == count,
        holding=3,
        seconds=30,
    )


def test_abort_returns_its_status_and_message(rig: Rig) -> None:
    """`v2_gate` returns `Abort(status=451, message="blocked by the cblv2 v2 gate")` at pre_call. The caller must
    get HTTP 451 with `error.message` equal to that message (proxy/utils.py v2_abort_error builds
    HTTPException(451, {"error": message}))."""
    response: Final = chat(rig, f"{V2_ABORT} {marker()}")
    assert response.status_code == V2_ABORT_STATUS, response.text
    assert error_message(response) == V2_ABORT_MESSAGE


def test_abort_headers_reach_the_caller(rig: Rig) -> None:
    """The Abort carries `headers={"x-cblv2-blocked-by": "cblv2_v2_gate"}`; the 451 response must carry that
    header with that value."""
    response: Final = chat(rig, f"{V2_ABORT} {marker()}")
    assert response.status_code == V2_ABORT_STATUS
    assert response.headers.get(V2_ABORT_HEADER) == "cblv2_v2_gate"


def test_abort_never_reaches_the_provider(rig: Rig) -> None:
    """An aborted request must not be sent upstream: the provider receives no request carrying its text."""
    text: Final = f"{V2_ABORT} {marker()}"
    rig.provider_texts()
    assert chat(rig, text).status_code == V2_ABORT_STATUS
    assert rig.provider_texts() == ()


def test_abort_stops_the_legacy_logger_listed_after_it(rig: Rig) -> None:
    """`chain_first` is listed after `v2_gate`. When the gate aborts, chain_first's `async_pre_call_hook` must not
    run for that request: "If a serial or parallel handler returns Abort, nothing after it runs"."""
    text: Final = f"{V2_ABORT} {marker()}"
    assert chat(rig, text).status_code == V2_ABORT_STATUS
    held(
        lambda: rig.calls(text),
        lambda calls: ("v2_gate", "pre_call") in calls and ("chain_first", "async_pre_call_hook") not in calls,
        holding=3,
    )


def test_abort_runs_the_legacy_failure_hook(rig: Rig) -> None:
    """An aborted request is a failed request, so the legacy `async_post_call_failure_hook` of chain_first must run
    once for it, with the HTTPException the abort became. UNVERIFIED: read from raise_v2_abort and the endpoint's
    exception path in common_request_processing; not run."""
    text: Final = f"{V2_ABORT} {marker()}"
    assert chat(rig, text).status_code == V2_ABORT_STATUS
    found: Final = eventually(
        lambda: [r for r in rig.mentioning(text, "chain_first") if r["hook"] == "async_post_call_failure_hook"],
        bool,
        seconds=20,
    )
    assert len(found) == 1
    assert found[0]["error"] == "HTTPException"


def test_abort_with_its_own_exception_raises_that_exception(rig: Rig) -> None:
    """`Abort(status=..., message=..., exception=HTTPException(409, {"error": "conflict raised by the cblv2 v2
    gate"}))` must answer with the exception's status 409 and message, not with a status built from the Abort."""
    response: Final = chat(rig, f"{V2_ABORT_CUSTOM_EXCEPTION} {marker()}")
    assert response.status_code == V2_EXCEPTION_STATUS, response.text
    assert error_message(response) == V2_EXCEPTION_MESSAGE


def test_abort_of_a_streamed_request_is_a_plain_error_before_any_chunk(rig: Rig) -> None:
    """A streamed request the gate aborts must be answered 451 with a JSON error body and no SSE `data:` frames."""
    status, headers, body = stream_chat(rig, f"{V2_ABORT} {marker()}")
    assert status == V2_ABORT_STATUS
    assert "data: " not in body
    assert headers.get("content-type", "").startswith("application/json")


def test_abort_is_logged_as_one_failed_attempt(rig: Rig) -> None:
    """An aborted request must produce exactly one `attempt_ended` record with outcome "failure" on `v2_reader`
    and none with outcome "success". UNVERIFIED: a pre-call rejection is logged through the proxy-only failure
    path (_handle_logging_proxy_only_error); that this reaches V2 attempt_ended was not traced."""
    text: Final = f"{V2_ABORT} {marker()}"
    assert chat(rig, text).status_code == V2_ABORT_STATUS
    records: Final = _attempts(rig, text, "v2_reader", 1)
    assert records[0]["outcome"] == "failure"


def test_messages_patch_replaces_the_request_the_provider_receives(rig: Rig) -> None:
    """`v2_gate` returns `Patch().set(Field.messages, [{"role": "user", "content": "rewritten by the cblv2 v2
    gate"}])`. The provider must receive exactly that text as the last user message and the caller still gets
    200 with the scripted reply."""
    rig.provider_texts()
    response: Final = chat(rig, f"{V2_PATCH} {marker()}")
    assert response.status_code == 200, response.text
    assert reply_text(response) == REPLY_TEXT
    assert rig.provider_texts() == (V2_REWRITTEN_REQUEST,)


def test_legacy_logger_after_a_v2_writer_sees_the_patched_messages(rig: Rig) -> None:
    """chain_first is listed after v2_gate, so its `async_pre_call_hook` must see the patched messages, not the
    caller's text. Matched by `x-litellm-call-id`, because the caller's text is gone after the patch."""
    call_id: Final = f"cblv2-{marker()}"
    assert chat(rig, f"{V2_PATCH} {marker()}", headers={"x-litellm-call-id": call_id}).status_code == 200
    found: Final = eventually(
        lambda: [
            record
            for record in rig.probe.records()
            if record.get("litellm_call_id") == call_id
            and record.get("logger") == "chain_first"
            and record.get("hook") == "async_pre_call_hook"
        ],
        bool,
        seconds=20,
    )
    assert V2_REWRITTEN_REQUEST in str(found[0]["text"])


def test_exception_in_a_pre_call_handler_fails_the_request(rig: Rig) -> None:
    """A V2 pre_call handler that raises RuntimeError must fail the request with HTTP 500, and the provider gets
    nothing: only the three failure-side hooks swallow handler exceptions."""
    text: Final = f"{V2_RAISE} {marker()}"
    rig.provider_texts()
    assert chat(rig, text).status_code == 500
    assert rig.provider_texts() == ()


def test_response_headers_patch_reaches_the_caller_on_success(rig: Rig) -> None:
    """`v2_gate` merges `x-cblv2-v2-gate: seen` into response_headers on every request; a successful chat's
    response must carry that header."""
    response: Final = chat(rig, marker())
    assert response.status_code == 200
    assert response.headers.get(V2_RESPONSE_HEADER) == "seen"


def test_response_headers_patch_reaches_the_caller_on_a_provider_error(rig: Rig) -> None:
    """The response_headers hook also runs when the provider answers 400, so the 400 must carry
    `x-cblv2-v2-gate: seen` too."""
    response: Final = chat(rig, marker(), model=CHAT_REJECTED)
    assert response.status_code == 400
    assert response.headers.get(V2_RESPONSE_HEADER) == "seen"


def test_response_headers_patch_reaches_a_streamed_response(rig: Rig) -> None:
    """A streamed chat must carry `x-cblv2-v2-gate: seen` in its response headers, which are sent before the first
    chunk."""
    status, headers, _ = stream_chat(rig, marker())
    assert status == 200
    assert headers.get(V2_RESPONSE_HEADER) == "seen"


def test_post_call_success_patch_replaces_the_reply(rig: Rig) -> None:
    """`v2_post_gate` sets the response's first choice content to "rewritten by the cblv2 v2 post-call handler"
    and returns `Patch().set(Field.response, response)`. The caller must receive that content. UNVERIFIED: the
    proxy post_call_success site maps Field.response back onto the returned object; not run."""
    response: Final = chat(rig, f"{V2_POST_PATCH} {marker()}")
    assert response.status_code == 200, response.text
    assert reply_text(response) == V2_REWRITTEN_RESPONSE


def test_attempt_ended_reader_gets_one_success_with_usage_and_cost(rig: Rig) -> None:
    """One chat must produce exactly one `attempt_ended` record on `v2_reader` with outcome "success",
    total_tokens 15 and response_cost 0.019, read from the typed `usage` and `cost` fields."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    record: Final = _attempts(rig, text, "v2_reader", 1)[0]
    assert record["outcome"] == "success"
    assert record["total_tokens"] == 15
    assert record["response_cost"] == pytest.approx(CHAT_COST)


def test_attempt_ended_reader_gets_one_failure_on_a_provider_error(rig: Rig) -> None:
    """A chat the provider answers with 400 must produce exactly one `attempt_ended` record on `v2_reader` with
    outcome "failure"."""
    text: Final = marker()
    assert chat(rig, text, model=CHAT_REJECTED).status_code == 400
    assert _attempts(rig, text, "v2_reader", 1)[0]["outcome"] == "failure"


def test_attempt_ended_fires_once_per_attempt_on_a_fallback(rig: Rig) -> None:
    """The primary deployment answers 500 and the router falls back to "cblv2-chat". `attempt_ended` is per
    attempt, so `v2_reader` must get two records: one "failure" and one "success". A port to `call_ended` would
    give one (s3-log-v3 spec decision D4)."""
    text: Final = marker()
    assert chat(rig, text, model=CHAT_FALLBACK_PRIMARY).status_code == 200
    outcomes: Final = sorted(str(record["outcome"]) for record in _attempts(rig, text, "v2_reader", 2))
    assert outcomes == ["failure", "success"]


def test_attempt_ended_fires_once_after_a_stream(rig: Rig) -> None:
    """A streamed chat read to the end must produce exactly one `attempt_ended` record with outcome "success" and
    total_tokens 15 for the whole stream, never one per chunk."""
    text: Final = marker()
    assert stream_chat(rig, text)[0] == 200
    record: Final = _attempts(rig, text, "v2_reader", 1)[0]
    assert record["outcome"] == "success"
    assert record["total_tokens"] == 15


def test_attempt_ended_model_is_the_deployment_model(rig: Rig) -> None:
    """The `model` field the reader sees on attempt_ended for a chat to "cblv2-chat" must be "gpt-4o-mini", the
    provider model of the deployment, not the model group. UNVERIFIED: Field.model on logging hooks maps
    kwargs["model"] (LOGGING_KEYS in litellm_logging.py)."""
    text: Final = marker()
    assert chat(rig, text, model=CHAT).status_code == 200
    assert _attempts(rig, text, "v2_reader", 1)[0]["model"] == "gpt-4o-mini"


def test_default_selection_records_no_log_requests(rig: Rig) -> None:
    """`v2_reader` has no Selection, so `honors_no_log` is False and a `no-log: true` chat must still produce one
    `attempt_ended` record on it. The guide calls this out as a migration difference from legacy loggers, which
    are skipped; it is pinned so it changes only on purpose."""
    text: Final = marker()
    assert chat(rig, text, **{"no-log": True}).status_code == 200
    assert len(_attempts(rig, text, "v2_reader", 1)) == 1


def test_honors_no_log_skips_the_reader_for_no_log_requests(rig: Rig) -> None:
    """`v2_no_log_reader` declares `honors_no_log=True`, so a `no-log: true` chat must produce no record on it,
    while a normal chat produces one."""
    skipped: Final = marker()
    logged: Final = marker()
    assert chat(rig, skipped, **{"no-log": True}).status_code == 200
    assert chat(rig, logged).status_code == 200
    _attempts(rig, logged, "v2_no_log_reader", 1)
    assert _attempts(rig, skipped, "v2_no_log_reader", 0) == ()


def test_no_log_skips_the_legacy_logger_but_not_its_proxy_hooks(rig: Rig) -> None:
    """For a `no-log: true` chat, the legacy `chain_first` must still run its pre-call hook (proxy hooks are not
    logging) but get no `async_log_success_event`."""
    text: Final = marker()
    assert chat(rig, text, **{"no-log": True}).status_code == 200
    _attempts(rig, text, "v2_reader", 1)
    calls: Final = rig.calls(text, "chain_first")
    assert ("chain_first", "async_pre_call_hook") in calls
    assert ("chain_first", "async_log_success_event") not in calls


def test_caller_opt_out_header_skips_an_opted_in_reader(rig: Rig) -> None:
    """On a licensed proxy, `x-litellm-disable-callbacks: cblv2_no_log_reader` must skip `v2_no_log_reader`
    (it allows OptOutBy.caller) for that request, while `v2_reader` still records it."""
    if not LICENSED:
        pytest.skip("dynamic callback disabling is honoured only with LITELLM_LICENSE set")
    text: Final = marker()
    assert chat(rig, text, headers={"x-litellm-disable-callbacks": "cblv2_no_log_reader"}).status_code == 200
    _attempts(rig, text, "v2_reader", 1)
    assert _attempts(rig, text, "v2_no_log_reader", 0) == ()


def test_caller_cannot_opt_out_of_a_blocking_logger(rig: Rig) -> None:
    """On a licensed proxy, `x-litellm-disable-callbacks: cblv2_v2_gate` must not remove `v2_gate`, which writes
    and may abort: an abort directive is still answered 451. build_manifest refuses caller opt-out for blocking
    loggers, so the header has nothing to match."""
    if not LICENSED:
        pytest.skip("dynamic callback disabling is honoured only with LITELLM_LICENSE set")
    response: Final = chat(rig, f"{V2_ABORT} {marker()}", headers={"x-litellm-disable-callbacks": "cblv2_v2_gate"})
    assert response.status_code == V2_ABORT_STATUS


def test_crashing_reader_never_fails_the_request_or_other_readers(rig: Rig) -> None:
    """`v2_crashing_reader` raises on every attempt_ended. A chat must still return 200 with the scripted reply,
    and `v2_reader` and `v2_awaited_reader` still record the attempt."""
    text: Final = marker()
    response: Final = chat(rig, text)
    assert response.status_code == 200
    assert reply_text(response) == REPLY_TEXT
    _attempts(rig, text, "v2_reader", 1)
    _attempts(rig, text, "v2_awaited_reader", 1)


def test_awaited_reader_is_called_once_with_the_outcome(rig: Rig) -> None:
    """`v2_awaited_reader` (`awaited=True`) must be called exactly once per attempt with outcome "success" for a
    successful chat."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    assert _attempts(rig, text, "v2_awaited_reader", 1)[0]["outcome"] == "success"


def test_v2_and_legacy_loggers_each_see_one_event_per_call(rig: Rig) -> None:
    """One chat must give the legacy chain_first exactly one `async_log_success_event` and the V2 v2_reader exactly
    one `attempt_ended`: the legacy adapter and the V2 dispatcher never deliver the same event twice."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    _attempts(rig, text, "v2_reader", 1)
    held(
        lambda: [hook for _, hook in rig.calls(text, "chain_first") if hook == "async_log_success_event"],
        lambda hooks: len(hooks) == 1,
        holding=3,
    )


def test_metadata_patch_at_pre_call_is_applied(rig: Rig) -> None:
    """A V2 pre_call handler that declares `writes={Field.metadata}` and returns
    `Patch().merge(Field.metadata, {"cblv2": "patched"})` should let the request succeed with the metadata
    merged. Today the proxy raises InvalidPatchError ("metadata has no place in the legacy kwargs of this hook")
    when it applies the patch, per the user guide's Unsettled table."""
    pytest.skip("BUG: pre_call metadata patches raise InvalidPatchError on the proxy instead of being applied")
    response: Final = chat(rig, f"{V2_METADATA_PATCH} {marker()}")
    assert response.status_code == 200, response.text


def test_v2_gate_sees_every_request_once(rig: Rig) -> None:
    """A plain chat must call `v2_gate`'s pre_call handler exactly once and the request must succeed unchanged:
    the provider receives the caller's text."""
    text: Final = marker()
    rig.provider_texts()
    assert chat(rig, text, model=CHAT).status_code == 200
    held(
        lambda: [hook for _, hook in rig.calls(text, "v2_gate") if hook == "pre_call"],
        lambda hooks: len(hooks) == 1,
        holding=2,
    )
    assert rig.provider_texts() == (text,)
