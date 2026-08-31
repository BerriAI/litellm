from typing import Final

from tests.transform_contracts.registry import run_contract_case
from tests.transform_contracts.schema import JsonValue, TransformationCase, expected_output


def test_transformation_contract(contract_case: TransformationCase) -> None:
    actual: Final[JsonValue] = run_contract_case(contract_case)
    expected: Final[JsonValue] = expected_output(contract_case)
    assert actual == expected
