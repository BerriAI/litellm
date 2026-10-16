from __future__ import annotations

from typing import Final

from setup.event_loop import EventLoopStalls, KnownStall, offenders, parse_stalls, summarize

_HANDLER: Final = (
    "Executing <Task pending name='Task-9' coro=<RequestResponseCycle.run_asgi() running at "
    "/usr/local/lib/python3.12/site-packages/uvicorn/protocols/http/h11_impl.py:416> wait_for=<Future pending "
    "cb=[Task.task_wakeup()]> created at /usr/local/lib/python3.12/asyncio/base_events.py:448> took 0.731 seconds"
)
_STREAM: Final = (
    "Executing <Task pending name='Task-31' coro=<TaskHandle._run_coro() running at "
    "/usr/local/lib/python3.12/site-packages/anyio/_core/_tasks.py:275> created at "
    "/usr/local/lib/python3.12/asyncio/tasks.py:420> took 0.412 seconds"
)
_STARTUP: Final = (
    "Executing <Task pending name='Task-2' coro=<LifespanOn.main() running at "
    "/usr/local/lib/python3.12/site-packages/uvicorn/lifespan/on.py:86> created at "
    "/usr/local/lib/python3.12/asyncio/base_events.py:448> took 0.302 seconds"
)
_SPEC: Final = EventLoopStalls(
    allowlist=(KnownStall("LifespanOn.main", max_count=3), KnownStall("RequestResponseCycle.run_asgi", max_count=2))
)


def test_parse_keeps_the_coroutine_its_location_and_the_hold() -> None:
    (stall,) = parse_stalls(("INFO: plain log line", _HANDLER))
    assert stall.coroutine == "RequestResponseCycle.run_asgi()"
    assert stall.where == "uvicorn/protocols/http/h11_impl.py:416"
    assert stall.seconds == 0.731


def test_a_location_off_the_allowlist_offends_once_however_often_it_appears() -> None:
    rows: Final = summarize(_SPEC, parse_stalls((_STARTUP, _STREAM, _STREAM, _HANDLER)))
    assert [(row.coroutine, row.count, row.offending) for row in rows] == [
        ("TaskHandle._run_coro()", 2, True),
        ("RequestResponseCycle.run_asgi()", 1, False),
        ("LifespanOn.main()", 1, False),
    ]
    assert offenders(rows) == (
        "TaskHandle._run_coro() anyio/_core/_tasks.py:275: 2x, max 0.41 s (not on the allowlist)",
    )


def test_a_known_location_offends_only_past_its_count() -> None:
    within: Final = summarize(_SPEC, parse_stalls((_HANDLER, _HANDLER)))
    assert offenders(within) == ()
    past: Final = summarize(_SPEC, parse_stalls((_HANDLER, _HANDLER, _HANDLER)))
    assert offenders(past) == (
        "RequestResponseCycle.run_asgi() uvicorn/protocols/http/h11_impl.py:416: 3x, max 0.73 s "
        "(allowlist RequestResponseCycle.run_asgi permits 2)",
    )


def test_holds_under_the_threshold_are_not_rows() -> None:
    short: Final = _HANDLER.replace("took 0.731 seconds", "took 0.099 seconds")
    assert summarize(_SPEC, parse_stalls((short,))) == ()
