from __future__ import annotations

import time
from typing import Final

import pytest

from tests.integration._support.timing import (
    assert_cut_short,
    assert_not_delayed_by,
    assert_within_factor_of_control,
    timed,
)


def test_timed_returns_the_value_and_at_least_the_time_the_operation_took() -> None:
    measured: Final = timed(lambda: time.sleep(0.05) or "done")
    assert measured.value == "done"
    assert measured.seconds >= 0.05


def test_a_burst_slowed_by_load_but_not_by_the_injected_delay_passes() -> None:
    assert_not_delayed_by((1.2, 2.4), control=(1.1, 1.9), injected_delay=2)


def test_a_burst_that_waited_out_the_injected_delay_fails() -> None:
    with pytest.raises(AssertionError, match="injected delay"):
        assert_not_delayed_by((1.2, 4.0), control=(1.1, 1.9), injected_delay=2)


def test_a_call_cut_short_before_the_injected_stall_passes_and_one_that_waited_fails() -> None:
    assert_cut_short(1.5, injected_delay=20)
    with pytest.raises(AssertionError, match="no sooner than"):
        assert_cut_short(20.0, injected_delay=20)


def test_a_bound_relative_to_the_control_scales_with_a_slower_machine() -> None:
    assert_within_factor_of_control(0.9, control=0.1, factor=10)
    assert_within_factor_of_control(9.0, control=1.0, factor=10)
    with pytest.raises(AssertionError, match="over 10x"):
        assert_within_factor_of_control(1.1, control=0.1, factor=10)
