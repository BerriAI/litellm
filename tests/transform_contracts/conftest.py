from __future__ import annotations

from typing import Final

import pytest

from tests.transform_contracts.loader import load_contract_cases


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "contract_case" not in metafunc.fixturenames:
        return
    cases: Final = load_contract_cases()
    metafunc.parametrize("contract_case", cases, ids=tuple(case.id for case in cases))


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
