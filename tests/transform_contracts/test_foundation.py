from pathlib import Path
from typing import Final

import pytest

from tests.transform_contracts.loader import discover_contract_paths, load_contract_cases

_VALID_CASE: Final = """
{
  "schema_version": 1,
  "cases": [
    {
      "id": "mistral.ocr.get_supported_ocr_params.latest",
      "operation": "mistral.ocr.get_supported_ocr_params",
      "input": {"model": "mistral-ocr-latest"},
      "expected": ["pages"]
    }
  ]
}
"""


def _write(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def test_contract_discovery_is_sorted(tmp_path: Path) -> None:
    first: Final = tmp_path / "a.json"
    second: Final = tmp_path / "nested" / "b.json"
    second.parent.mkdir()
    _write(second, _VALID_CASE)
    _write(first, _VALID_CASE)
    assert discover_contract_paths(tmp_path) == (first, second)


def test_invalid_json_fails_loudly(tmp_path: Path) -> None:
    _write(tmp_path / "invalid.json", "{")
    with pytest.raises(ValueError, match="invalid transformation contract file"):
        load_contract_cases(tmp_path)


def test_unsupported_schema_version_fails_loudly(tmp_path: Path) -> None:
    _write(tmp_path / "future.json", _VALID_CASE.replace('"schema_version": 1', '"schema_version": 2'))
    with pytest.raises(ValueError, match="invalid transformation contract file"):
        load_contract_cases(tmp_path)


def test_duplicate_case_ids_fail_loudly(tmp_path: Path) -> None:
    _write(tmp_path / "first.json", _VALID_CASE)
    _write(tmp_path / "second.json", _VALID_CASE)
    with pytest.raises(ValueError, match="duplicate transformation contract case ids"):
        load_contract_cases(tmp_path)


def test_missing_required_field_fails_loudly(tmp_path: Path) -> None:
    _write(
        tmp_path / "missing.json",
        _VALID_CASE.replace('"input": {"model": "mistral-ocr-latest"},', '"input": {},'),
    )
    with pytest.raises(ValueError, match="invalid transformation contract file"):
        load_contract_cases(tmp_path)


def test_unsupported_operation_fails_loudly(tmp_path: Path) -> None:
    _write(
        tmp_path / "unsupported.json",
        _VALID_CASE.replace("mistral.ocr.get_supported_ocr_params", "mistral.ocr.unsupported"),
    )
    with pytest.raises(ValueError, match="invalid transformation contract file"):
        load_contract_cases(tmp_path)


def test_case_id_must_match_operation_namespace(tmp_path: Path) -> None:
    _write(
        tmp_path / "unstable-id.json",
        _VALID_CASE.replace("mistral.ocr.get_supported_ocr_params.latest", "mistral.ocr.other.latest"),
    )
    with pytest.raises(ValueError, match="invalid transformation contract file"):
        load_contract_cases(tmp_path)
