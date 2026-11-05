from __future__ import annotations

import math
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter
from setup.event_loop import EventLoopStalls, KnownStall, junit_properties, offenders, parse_stalls, summarize

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
_GC_PASS: Final = (
    "2026-10-17 04:46:02,101 - LiteLLM Proxy - WARNING - Garbage collection gen=2 collected=98608 took 0.711 seconds"
)
_SPEC: Final = EventLoopStalls(
    allowlist=(
        KnownStall("LifespanOn.main", max_count=3),
        KnownStall("RequestResponseCycle.run_asgi", max_count=2),
        KnownStall("garbage collection", max_count=1),
    )
)


class _Slice(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    value: float
    share: float | None
    note: str | None
    over: bool


def _breakdown(properties: dict[str, float | str], name: str) -> list[_Slice]:
    return TypeAdapter(list[_Slice]).validate_json(str(properties[f"perf_event_loop_{name}_breakdown"]))


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


def test_a_hold_the_collector_pass_before_it_accounts_for_is_the_collectors_not_the_tasks() -> None:
    rows: Final = summarize(_SPEC, parse_stalls((_GC_PASS, _STREAM.replace("0.412", "0.731"))))
    assert [(row.coroutine, row.where, row.max_seconds, row.offending) for row in rows] == [
        ("garbage collection", "gen=2", 0.731, False)
    ]


def test_a_hold_well_past_its_collector_pass_stays_on_the_task() -> None:
    short_pass: Final = _GC_PASS.replace("0.711", "0.200")
    rows: Final = summarize(_SPEC, parse_stalls((short_pass, _STREAM.replace("0.412", "0.731"))))
    assert [(row.coroutine, row.offending) for row in rows] == [("TaskHandle._run_coro()", True)]


def test_a_hold_is_charged_to_the_longest_collector_pass_that_fits_it_not_to_their_sum() -> None:
    thread_side_pass: Final = _GC_PASS.replace("0.711", "0.627")
    rows: Final = summarize(
        _SPEC, parse_stalls((thread_side_pass, _GC_PASS.replace("0.711", "0.605"), _STREAM.replace("0.412", "0.614")))
    )
    assert [(row.coroutine, row.where, row.max_seconds, row.offending) for row in rows] == [
        ("garbage collection", "gen=2", 0.614, False)
    ]


def test_a_collector_pass_explains_only_the_next_hold_and_only_one_at_least_as_long() -> None:
    rows: Final = summarize(_SPEC, parse_stalls((_GC_PASS, _STARTUP, _HANDLER)))
    assert sorted((row.coroutine, row.offending) for row in rows) == [
        ("LifespanOn.main()", False),
        ("RequestResponseCycle.run_asgi()", False),
    ]
    assert summarize(_SPEC, parse_stalls((_GC_PASS,))) == ()


def test_junit_properties_count_every_hold_and_only_the_holds_past_their_allowance_as_offending() -> None:
    rows: Final = summarize(_SPEC, parse_stalls((_STARTUP, _STREAM, _STREAM, _HANDLER, _HANDLER, _HANDLER)))
    properties: Final = dict(junit_properties(_SPEC, rows))
    assert properties["perf_event_loop_holds"] == 6
    assert properties["perf_event_loop_holds_limit"] == 6
    assert properties["perf_event_loop_offending_holds"] == 3
    assert properties["perf_event_loop_offending_holds_limit"] == 0
    assert properties["perf_event_loop_hold_seconds_max"] == 0.731
    assert math.isclose(float(properties["perf_event_loop_hold_seconds_total"]), 0.302 + 2 * 0.412 + 3 * 0.731)
    breakdown: Final = _breakdown(properties, "holds")
    assert [(slice.label, slice.value, slice.note, slice.over) for slice in breakdown] == [
        (
            "RequestResponseCycle.run_asgi() uvicorn/protocols/http/h11_impl.py:416",
            3,
            "3 of 2 allowed, max 0.73 s",
            True,
        ),
        ("TaskHandle._run_coro() anyio/_core/_tasks.py:275", 2, "not on the allowlist, max 0.41 s", True),
        ("LifespanOn.main() uvicorn/lifespan/on.py:86", 1, "1 of 3 allowed, max 0.30 s", False),
    ]
    assert math.isclose(sum(slice.share or 0.0 for slice in breakdown), 1.0)


def test_the_offending_breakdown_lists_only_the_locations_past_their_cap_by_how_far_over_they_are() -> None:
    rows: Final = summarize(_SPEC, parse_stalls((_STARTUP, _STREAM, _STREAM, _HANDLER, _HANDLER, _HANDLER)))
    offending: Final = _breakdown(dict(junit_properties(_SPEC, rows)), "offending_holds")
    assert [(slice.label, slice.value, slice.share, slice.over) for slice in offending] == [
        ("RequestResponseCycle.run_asgi() uvicorn/protocols/http/h11_impl.py:416", 1, 1 / 3, True),
        ("TaskHandle._run_coro() anyio/_core/_tasks.py:275", 2, 2 / 3, True),
    ]
    clean: Final = dict(junit_properties(_SPEC, summarize(_SPEC, parse_stalls((_HANDLER, _HANDLER)))))
    assert clean["perf_event_loop_offending_holds"] == 0
    assert "perf_event_loop_offending_holds_breakdown" not in clean


def test_the_seconds_breakdowns_carry_the_time_per_location_and_the_longest_single_holds() -> None:
    long_handler: Final = _HANDLER.replace("0.731", "0.950")
    rows: Final = summarize(_SPEC, parse_stalls((_STARTUP, _STREAM, long_handler, _HANDLER)))
    properties: Final = dict(junit_properties(_SPEC, rows))
    per_location: Final = _breakdown(properties, "hold_seconds_total")
    assert [(slice.label.split()[0], slice.value) for slice in per_location] == [
        ("RequestResponseCycle.run_asgi()", 0.950 + 0.731),
        ("TaskHandle._run_coro()", 0.412),
        ("LifespanOn.main()", 0.302),
    ]
    assert math.isclose(sum(slice.share or 0.0 for slice in per_location), 1.0)
    longest: Final = _breakdown(properties, "hold_seconds_max")
    assert [(slice.label.split()[0], slice.value, slice.share, slice.over) for slice in longest] == [
        ("RequestResponseCycle.run_asgi()", 0.950, None, False),
        ("RequestResponseCycle.run_asgi()", 0.731, None, False),
        ("TaskHandle._run_coro()", 0.412, None, True),
        ("LifespanOn.main()", 0.302, None, False),
    ]


def test_the_longest_holds_breakdown_stops_at_ten() -> None:
    rows: Final = summarize(_SPEC, parse_stalls(tuple(_STARTUP for _ in range(12))))
    longest: Final = _breakdown(dict(junit_properties(_SPEC, rows)), "hold_seconds_max")
    assert len(longest) == 10


def test_junit_properties_without_holds_report_zeros_and_no_breakdown() -> None:
    properties: Final = dict(junit_properties(_SPEC, ()))
    assert properties["perf_event_loop_holds"] == 0
    assert properties["perf_event_loop_offending_holds"] == 0
    assert properties["perf_event_loop_hold_seconds_max"] == 0
    assert not [name for name in properties if name.endswith("_breakdown")]
