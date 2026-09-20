"""The parity diff itself: what it reports is what keeps the allow-lists and the gap ledger honest."""

from types import MappingProxyType
from typing import Final

from litellm.callbacks_v1.builtin.manifest import Gap
from tests.test_litellm.callbacks_v1.builtin.support import Allowed, unexplained

COST: Final = Gap("fact", "cost", "kwargs['response_cost']", "no cost fact")
LEGACY: Final = MappingProxyType({"id": "call-1", "data": MappingProxyType({"cost": 0.25, "model": "claude"})})
PORT: Final = MappingProxyType({"id": "call-1", "data": MappingProxyType({"cost": None, "model": "claude"})})


def test_identical_bodies_need_no_allowance() -> None:
    assert unexplained(LEGACY, LEGACY, (), ()) == ()


def test_a_difference_nobody_allowed_is_reported_by_path() -> None:
    assert unexplained(LEGACY, PORT, (), ()) == ("differs: data.cost: legacy 0.25 != port None",)


def test_a_difference_allowed_by_an_open_gap_is_parity() -> None:
    assert unexplained(LEGACY, PORT, (Allowed("data.cost", "no cost fact", gap="cost"),), (COST,)) == ()


def test_an_allowance_outlives_neither_its_difference_nor_its_gap() -> None:
    allowed: Final = (Allowed("data.cost", "no cost fact", gap="cost"),)

    assert unexplained(LEGACY, LEGACY, allowed, (COST,)) == (
        "stale Allowed('data.cost'): it no longer matches any difference",
    )
    assert unexplained(LEGACY, PORT, allowed, ()) == (
        "Allowed('data.cost') cites gap 'cost', which the port does not list",
    )
