from __future__ import annotations

import importlib

contracts = importlib.import_module("tests.rust-python-harness.shared.unit_runners.contracts")

UNIT_TEST_CONTRACTS = contracts.UNIT_TEST_CONTRACTS


def test_should_leave_functions_without_unit_test_contracts_unimplemented() -> None:
    assert "messages" not in UNIT_TEST_CONTRACTS
