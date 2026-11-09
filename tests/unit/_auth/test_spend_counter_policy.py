from dataclasses import replace
from typing import Final

import pytest

from litellm._auth import spend_counter_policy as core

_SPEND: Final = core.Spend(2.0, False, 0.0, 10.0, False, False)


@pytest.mark.parametrize(
    "facts, expected",
    (
        (_SPEND, core.Action("ReadStrictBefore")),
        (replace(_SPEND, limit=None), core.SpendValue("Value", 2.0)),
        (replace(_SPEND, current=10.0), core.SpendValue("Value", 10.0)),
        (replace(_SPEND, current=11.0), core.SpendValue("Value", 11.0)),
        (replace(_SPEND, fallback=3.0), core.Action("ReadFloor")),
        (replace(_SPEND, window=True), core.Action("ReadFloor")),
        (replace(_SPEND, strict_before=True), core.Action("ReadFloor")),
        (replace(_SPEND, strict_before=False), core.Action("ReadStrictAfter")),
        (replace(_SPEND, strict_before=False, verified=True), core.SpendValue("Value", 2.0)),
        (replace(_SPEND, strict_before=False, fallback_authoritative=True), core.SpendValue("Value", 2.0)),
        (replace(_SPEND, strict_before=True, authoritative_read=True), core.Action("ReadStrictAfter")),
        (
            replace(_SPEND, strict_before=True, authoritative_read=True, authoritative=3.0),
            core.SpendValue("Repair", 3.0),
        ),
        (
            replace(_SPEND, strict_before=True, authoritative_read=True, authoritative=1.0),
            core.SpendValue("Value", 2.0),
        ),
        (replace(_SPEND, fallback=3.0, authoritative_read=True), core.SpendValue("Value", 3.0)),
        (replace(_SPEND, strict_before=False, strict_after=True), core.Action("Unverifiable")),
        (
            replace(_SPEND, strict_before=True, authoritative_read=True, strict_after=False),
            core.SpendValue("Value", 2.0),
        ),
    ),
)
def test_spend_selection_preserves_lazy_flag_reads(facts: core.Spend, expected: core.Decision) -> None:
    assert core.decide(facts) == expected


@pytest.mark.parametrize("current", (10.0 + 1e-13, 10.5, 11.0, 12.0, 13.0))
@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize("throttled", (False, True))
def test_overflow_matches_reference(current: float, strict: bool, throttled: bool) -> None:
    remaining: Final = 10.0 - (current - 2.0)
    expected: Final = (
        core.Action("Release")
        if throttled
        else core.Rejected("Reject", current, None)
        if remaining <= 1e-12
        else core.Rejected("Reject", current - 2.0, 2.0)
        if strict and current - 10.0 > 1e-12
        else core.Resize("Resize", remaining)
    )
    assert core.decide(core.OverBudget(throttled, current, 10.0, 2.0, strict)) == expected


@pytest.mark.parametrize("token_present", (False, True))
@pytest.mark.parametrize("token", (None, "", "hash"))
@pytest.mark.parametrize("counter", ("spend:key:hash", "spend:key:None", "spend:key:", "spend:team:hash"))
def test_throttle_is_limited_to_the_key_counter(token_present: bool, token: str | None, counter: str) -> None:
    expected: Final = "ReadThrottle" if token_present and counter == f"spend:key:{token}" else "Skip"
    assert core.decide(core.ThrottleKey(counter, token_present, token)) == core.Action(expected)


@pytest.mark.parametrize("currents, limits", (((), ()), ((0.0, 1.0), (2.0, 2.0)), ((0.0, 2.0), (2.0, 2.0))))
def test_batching_preserves_order_and_overflow_boundaries(
    currents: tuple[float, ...], limits: tuple[float, ...]
) -> None:
    expected: Final = (
        ()
        if not currents
        else (tuple(range(len(currents))),)
        if all(current + 1.0 <= limit for current, limit in zip(currents, limits, strict=True))
        else tuple((index,) for index in range(len(currents)))
    )
    assert core.decide(core.ReservationGroups(currents, limits, 1.0)) == core.Groups("ReservationGroups", expected)
