"""Generic custom callbacks from config run in config order on every hook, and that order must survive the move
to the CustomLoggerV2 planner, which orders callbacks from their declarations (step0/list-order.md).

Contract. With `callbacks: [chain_first, chain_second, chain_third]` (three instances of one legacy CustomLogger
class from `dec1_cblv2_probe`):
- `async_pre_call_hook` runs first, second, third, and each sees the request the previous one returned, so the
  provider receives the text with three tags appended in that order
- a hook that raises HTTPException stops the hooks after it, the provider gets nothing, and the caller gets the
  hook's status and message; a hook that crashes with a plain exception fails the request with 500
- `async_post_call_success_hook` runs in the same order and each sees the previous rewrite
- `async_post_call_streaming_iterator_hook` wraps in list order, so the last listed is outermost
- `async_log_success_event` and `async_log_failure_event` run in list order, and a crashing logger neither
  changes the response nor stops the loggers after it
- `async_post_call_failure_hook` runs for every listed logger in list order, including those whose pre-call hook
  never ran because an earlier one rejected the request

The legacy adapter keeps a legacy logger in its list position and treats it as writing everything, so it gets its
own serial step; these tests are what proves that.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import held
from integration.observability.dec1_cblv2_probe import (
    CHAIN,
    CHAIN_REJECTION,
    CRASH_LOG_AT,
    CRASH_PRE_AT,
    ITERATOR,
    REJECT_AT,
)
from integration.observability.dec1_cblv2_support import (
    CHAT_REJECTED,
    PROBE_MODULE,
    REPLY_TEXT,
    Rig,
    cblv2_rig,
    chat,
    error_message,
    marker,
    reply_text,
    stream_chat,
    streamed_text,
)

ORDER: Final = ("chain_first", "chain_second", "chain_third")


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("cblv2-callback-order")
    with cblv2_rig(directory, {"callbacks": [f"{PROBE_MODULE}.{name}" for name in ORDER]}) as booted:
        yield booted


def _loggers_for(rig: Rig, text: str, hook: str, count: int) -> tuple[str, ...]:
    """Names of the loggers that ran `hook` for the request carrying `text`, in call order, once `count` arrived
    and no more arrive for 2 seconds."""
    return held(
        lambda: tuple(logger for logger, name in rig.calls(text) if name == hook),
        lambda loggers: len(loggers) == count,
        holding=2,
        seconds=30,
    )


def test_pre_call_hooks_run_in_config_order(rig: Rig) -> None:
    """One chat must call `async_pre_call_hook` on chain_first, chain_second and chain_third, once each, in that
    order: config order is the pre_call order for `callbacks:` entries."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    assert _loggers_for(rig, text, "async_pre_call_hook", 3) == ORDER


def test_each_pre_call_hook_sees_the_previous_rewrite(rig: Rig) -> None:
    """With the chain directive each pre-call hook appends ` [<its name>]` to the last message. The provider must
    receive `<text> [chain_first] [chain_second] [chain_third]`, proving each hook got the dict the one before it
    returned."""
    text: Final = f"{CHAIN} {marker()}"
    rig.provider_texts()
    assert chat(rig, text).status_code == 200
    assert rig.provider_texts() == (f"{text} [chain_first] [chain_second] [chain_third]",)


def test_post_call_success_hooks_run_in_config_order(rig: Rig) -> None:
    """One chat must call `async_post_call_success_hook` on the three loggers once each, in config order."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    assert _loggers_for(rig, text, "async_post_call_success_hook", 3) == ORDER


def test_each_post_call_success_hook_sees_the_previous_rewrite(rig: Rig) -> None:
    """With the chain directive each post-call hook appends ` [<its name>]` to the reply. The caller must receive
    `cblv2 scripted reply [chain_first] [chain_second] [chain_third]`."""
    response: Final = chat(rig, f"{CHAIN} {marker()}")
    assert response.status_code == 200
    assert reply_text(response) == f"{REPLY_TEXT} [chain_first] [chain_second] [chain_third]"


def test_success_log_events_run_in_config_order(rig: Rig) -> None:
    """One chat must deliver `async_log_success_event` to the three loggers once each, in config order. The
    logging loop awaits each logger in turn today; a V2 port that moves legacy readers off-path may reorder them,
    which is a visible change for loggers that depend on each other's metadata writes."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    assert _loggers_for(rig, text, "async_log_success_event", 3) == ORDER


def test_failure_log_events_run_in_config_order(rig: Rig) -> None:
    """A chat the provider answers with 400 must deliver `async_log_failure_event` to the three loggers once each,
    in config order."""
    text: Final = marker()
    assert chat(rig, text, model=CHAT_REJECTED).status_code == 400
    assert _loggers_for(rig, text, "async_log_failure_event", 3) == ORDER


def test_post_call_failure_hooks_run_in_config_order(rig: Rig) -> None:
    """A provider 400 must call `async_post_call_failure_hook` on the three loggers once each, in config order."""
    text: Final = marker()
    assert chat(rig, text, model=CHAT_REJECTED).status_code == 400
    assert _loggers_for(rig, text, "async_post_call_failure_hook", 3) == ORDER


def test_whole_request_runs_pre_call_then_post_call_then_logging_per_logger(rig: Rig) -> None:
    """For chain_second alone, one chat must produce pre_call, then post_call_success, then log_success in that
    order: the request-path hooks finish before the logging plane runs."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    hooks: Final = held(
        lambda: tuple(name for logger, name in rig.calls(text) if logger == "chain_second"),
        lambda names: len(names) == 3,
        holding=2,
        seconds=30,
    )
    assert hooks == ("async_pre_call_hook", "async_post_call_success_hook", "async_log_success_event")


def test_rejection_by_the_second_hook_stops_the_third_and_the_provider(rig: Rig) -> None:
    """chain_second raises HTTPException(400, {"error": "rejected by the cblv2 chain logger chain_second"}). The
    caller must get 400 with exactly that message, chain_first and chain_second ran their pre-call hooks,
    chain_third never did, and the provider received nothing."""
    text: Final = f"{REJECT_AT}chain_second {marker()}"
    rig.provider_texts()
    response: Final = chat(rig, text)
    assert response.status_code == 400, response.text
    assert error_message(response) == f"{CHAIN_REJECTION} chain_second"
    assert _loggers_for(rig, text, "async_pre_call_hook", 2) == ("chain_first", "chain_second")
    assert rig.provider_texts() == ()


def test_rejection_still_runs_every_failure_hook_in_order(rig: Rig) -> None:
    """When chain_first rejects at pre-call, `async_post_call_failure_hook` must still run on all three loggers in
    config order, including the two whose pre-call hook never ran."""
    text: Final = f"{REJECT_AT}chain_first {marker()}"
    assert chat(rig, text).status_code == 400
    assert _loggers_for(rig, text, "async_post_call_failure_hook", 3) == ORDER


def test_rejection_is_logged_once_per_logger_as_a_failure(rig: Rig) -> None:
    """A pre-call rejection by chain_first must deliver exactly one `async_log_failure_event` to each logger and no
    success event to any."""
    text: Final = f"{REJECT_AT}chain_first {marker()}"
    assert chat(rig, text).status_code == 400
    assert sorted(_loggers_for(rig, text, "async_log_failure_event", 3)) == sorted(ORDER)
    assert [hook for _, hook in rig.calls(text) if hook == "async_log_success_event"] == []


def test_pre_call_crash_fails_closed_with_500_and_skips_later_hooks(rig: Rig) -> None:
    """chain_second raises ValueError at pre-call. The caller must get 500, chain_third's pre-call hook never runs,
    and the provider receives nothing: a crashing hook never lets the request through unchecked."""
    text: Final = f"{CRASH_PRE_AT}chain_second {marker()}"
    rig.provider_texts()
    assert chat(rig, text).status_code == 500
    assert _loggers_for(rig, text, "async_pre_call_hook", 2) == ("chain_first", "chain_second")
    assert rig.provider_texts() == ()


def test_crashing_success_logger_does_not_stop_the_loggers_after_it(rig: Rig) -> None:
    """chain_first raises in `async_log_success_event`. The caller still gets 200 with the scripted reply, and
    chain_second and chain_third still receive their success event."""
    text: Final = f"{CRASH_LOG_AT}chain_first {marker()}"
    response: Final = chat(rig, text)
    assert response.status_code == 200
    assert reply_text(response) == REPLY_TEXT
    assert _loggers_for(rig, text, "async_log_success_event", 3) == ORDER


def test_streaming_iterator_hooks_wrap_in_config_order(rig: Rig) -> None:
    """With the iterator directive each logger prefixes `[<its name>]` to the first content chunk it sees. The
    first listed wraps the provider stream and each later one wraps the previous, so the caller must read
    `[chain_third][chain_second][chain_first]cblv2 scripted reply`."""
    status, _, sse = stream_chat(rig, f"{ITERATOR} {marker()}")
    assert status == 200
    assert streamed_text(sse) == f"[chain_third][chain_second][chain_first]{REPLY_TEXT}"


def test_streaming_iterator_hook_is_called_once_per_logger_per_stream(rig: Rig) -> None:
    """A streamed chat must call `async_post_call_streaming_iterator_hook` exactly once on each of the three
    loggers, not once per chunk. The hooks are wrapped in config order (proxy/utils.py:5272), so the last listed is
    the outermost generator and its body starts first when the caller pulls: the probe records the starts in
    reverse config order, chain_third, chain_second, chain_first (CI run 44161684479)."""
    text: Final = marker()
    assert stream_chat(rig, text)[0] == 200
    assert _loggers_for(rig, text, "async_post_call_streaming_iterator_hook", 3) == tuple(reversed(ORDER))


def test_streamed_chat_skips_the_post_call_success_hook(rig: Rig) -> None:
    """A streamed chat never calls `async_post_call_success_hook` (streams use the iterator hook instead), but
    still delivers one success log event per logger."""
    text: Final = marker()
    assert stream_chat(rig, text)[0] == 200
    _loggers_for(rig, text, "async_log_success_event", 3)
    assert [hook for _, hook in rig.calls(text) if hook == "async_post_call_success_hook"] == []


def test_order_is_per_request_under_concurrency(rig: Rig) -> None:
    """Two chats sent back to back must each see pre-call order first, second, third: the order is a property of
    each request, not of the process. Uses distinct markers so their records never mix."""
    texts: Final = [marker() for _ in range(2)]
    for text in texts:
        assert chat(rig, text).status_code == 200
    for text in texts:
        assert _loggers_for(rig, text, "async_pre_call_hook", 3) == ORDER


def test_unknown_text_leaves_the_request_untouched(rig: Rig) -> None:
    """Without a directive the three loggers change nothing: the provider receives the caller's text exactly and
    the caller receives the scripted reply exactly. This is the control for the rewrite tests above."""
    text: Final = f"plain {uuid.uuid4().hex}"
    rig.provider_texts()
    response: Final = chat(rig, text)
    assert reply_text(response) == REPLY_TEXT
    assert rig.provider_texts() == (text,)
