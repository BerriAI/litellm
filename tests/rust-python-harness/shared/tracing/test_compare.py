from __future__ import annotations

import pytest

from .compare import Operation, compare_traces


@pytest.mark.parametrize(
    ("rust", "message"),
    (
        ((Operation("decode", 0, 1), Operation("send", 2, 3)), None),
        ((Operation("decode", 0, 1), Operation("send", 2, 3), Operation("send", 4, 5)), "call count differs"),
        ((Operation("send", 0, 1), Operation("decode", 2, 3)), "required order"),
        ((Operation("decode", 0, 4), Operation("send", 2, 3)), "required order"),
        ((Operation("decode", 0, 1), Operation("unknown", 2, 3)), "unmapped Rust"),
    ),
)
def test_compare_mapped_calls_and_required_completion_order(rust: tuple[Operation, ...], message: str | None) -> None:
    problems = compare_traces(
        (Operation("parse", 0, 1), Operation("request", 2, 3)),
        rust,
        {"parse": "decode", "request": "send"},
        (("parse", "request"),),
    )
    if message is None:
        assert problems == ()
    else:
        assert any(message in problem for problem in problems)


def test_missing_required_operations_and_ambiguous_mappings_fail() -> None:
    assert compare_traces((), (), {"parse": "decode"}, (("parse", "request"),))
    assert compare_traces((), (), {"parse": "decode", "request": "decode"}) == ("ambiguous Rust operation: decode",)
