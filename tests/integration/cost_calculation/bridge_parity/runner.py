from typing import Final

from integration._support.client import Gateway
from integration.cost_calculation.bridge_parity.case import BridgeParityTestCase
from integration.cost_calculation.stream_parity.runner import bill_legs


def assert_bridge_parity(case: BridgeParityTestCase, gateway: Gateway) -> None:
    billed: Final = bill_legs(case.id, (case.native, case.bridged), gateway)
    native: Final = billed[case.native.name].model_dump(include=set(case.compared_row_fields))
    bridged: Final = billed[case.bridged.name].model_dump(include=set(case.compared_row_fields))
    assert bridged == native, (case.id, billed[case.native.name], billed[case.bridged.name])
