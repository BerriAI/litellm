from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from tests.transform_contracts.schema import CONTRACT_SUITE_ADAPTER, ContractSuiteV1, TransformationCase

CONTRACTS_ROOT: Final = Path(__file__).resolve().parent
CASES_ROOT: Final = CONTRACTS_ROOT / "cases"


def discover_contract_paths(root: Path = CASES_ROOT) -> tuple[Path, ...]:
    if not root.is_dir():
        raise FileNotFoundError(f"transformation contract directory does not exist: {root}")
    paths: Final = tuple(sorted(root.rglob("*.json")))
    if paths:
        return paths
    raise FileNotFoundError(f"no transformation contract files found under: {root}")


def load_contract_file(path: Path) -> ContractSuiteV1:
    try:
        return CONTRACT_SUITE_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ValueError(f"invalid transformation contract file: {path}") from exc


def load_contract_cases(root: Path = CASES_ROOT) -> tuple[TransformationCase, ...]:
    cases: Final = tuple(case for path in discover_contract_paths(root) for case in load_contract_file(path).cases)
    duplicates: Final = tuple(
        sorted(case_id for case_id, count in Counter(case.id for case in cases).items() if count > 1)
    )
    if duplicates:
        raise ValueError(f"duplicate transformation contract case ids: {duplicates}")
    return tuple(sorted(cases, key=lambda case: case.id))
