from typing import Final, Protocol

import pytest

from tests.transform_contracts.loader import load_contract_cases
from tests.transform_contracts.registry import run_contract_case
from tests.transform_contracts.schema import JsonValue, TransformationCase, expected_output

_CONTRACT_CASES: Final = load_contract_cases()


class _ContractCaseRequest(Protocol):
    @property
    def param(self) -> TransformationCase: ...


@pytest.fixture(params=_CONTRACT_CASES, ids=tuple(case.id for case in _CONTRACT_CASES))
def contract_case(request: _ContractCaseRequest, monkeypatch: pytest.MonkeyPatch) -> TransformationCase:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    return request.param


def test_transformation_contract(contract_case: TransformationCase) -> None:
    actual: Final[JsonValue] = run_contract_case(contract_case)
    expected: Final[JsonValue] = expected_output(contract_case)
    assert actual == expected
