from typing import Final, Literal

import pytest

from litellm.proxy.management_helpers.team_member_budget_update import matches_member_budget_update


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("keep", ()), ("raise", (0, 50)), ("lower", (150,)), ("both", (0, 50, 150))),
)
def test_selects_only_unequal_permanent_amounts_in_the_requested_direction(
    mode: Literal["keep", "raise", "lower", "both"], expected: tuple[int, ...]
) -> None:
    selected: Final = tuple(
        amount
        for amount in (None, 0, 50, 100, 150)
        if matches_member_budget_update(amount=amount, duration="30d", target=100, target_duration="30d", mode=mode)
    )
    assert selected == expected


@pytest.mark.parametrize("mode", ("raise", "lower", "both"))
@pytest.mark.parametrize(("duration", "target_duration"), (("7d", "30d"), (None, "30d"), ("30d", None)))
def test_never_converts_amounts_across_reset_periods(
    mode: Literal["raise", "lower", "both"], duration: str | None, target_duration: str | None
) -> None:
    selected: Final = tuple(
        amount
        for amount in (0, 50, 150)
        if matches_member_budget_update(
            amount=amount, duration=duration, target=100, target_duration=target_duration, mode=mode
        )
    )
    assert selected == ()


@pytest.mark.parametrize(
    ("duration", "target_duration"), (("1mo", "30d"), ("weekly", "1w"), ("60m", "1h"), ("60s", "1m"), (None, None))
)
def test_accepts_equivalent_reset_schedules(duration: str | None, target_duration: str | None) -> None:
    assert matches_member_budget_update(
        amount=50, duration=duration, target=100, target_duration=target_duration, mode="raise"
    )
