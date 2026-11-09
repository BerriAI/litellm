from itertools import product

import pytest

from litellm._auth import admission
from tests.unit._auth import reference_policy


@pytest.mark.parametrize("facts", tuple(admission.CommonGate(*values) for values in product((False, True), repeat=10)))
def test_common_gate_has_the_reference_skip_priority(facts: admission.CommonGate) -> None:
    assert admission.common_plan(facts) == reference_policy.common_plan(facts)


@pytest.mark.parametrize("facts", tuple(admission.Reservation(*values) for values in product((False, True), repeat=5)))
def test_reservation_plan_preserves_billing_agent_fail_closed(facts: admission.Reservation) -> None:
    assert admission.reservation_plan(facts) == reference_policy.reservation_plan(facts)


@pytest.mark.parametrize("facts", tuple(admission.Ownership(*values) for values in product((False, True), repeat=4)))
def test_observation_never_transfers_request_ownership(facts: admission.Ownership) -> None:
    assert admission.ownership_plan(facts).owner == "Python"
