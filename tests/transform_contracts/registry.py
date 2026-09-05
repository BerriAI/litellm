from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from tests.transform_contracts.mistral_ocr import (
    run_get_supported_ocr_params,
    run_map_ocr_params,
    run_transform_ocr_request,
    run_transform_ocr_response,
)
from tests.transform_contracts.schema import JsonValue, TransformationCase

ContractOperation = Callable[[TransformationCase], JsonValue]

OPERATIONS: Final[Mapping[str, ContractOperation]] = MappingProxyType(
    {
        "mistral.ocr.get_supported_ocr_params": run_get_supported_ocr_params,
        "mistral.ocr.map_ocr_params": run_map_ocr_params,
        "mistral.ocr.transform_ocr_request": run_transform_ocr_request,
        "mistral.ocr.transform_ocr_response": run_transform_ocr_response,
    }
)


def run_contract_case(case: TransformationCase) -> JsonValue:
    operation: Final = OPERATIONS.get(case.operation)
    if operation is None:
        raise ValueError(f"unsupported transformation contract operation: {case.operation}")
    return operation(case)
