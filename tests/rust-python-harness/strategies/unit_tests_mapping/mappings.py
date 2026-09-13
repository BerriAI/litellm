from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ...shared.reporting.models import SdkFunction
from .cases.ocr import OCR_CONTRACT
from .contracts import UnitTestContract

UNIT_TEST_CONTRACTS: Final[Mapping[SdkFunction, UnitTestContract]] = MappingProxyType({"ocr": OCR_CONTRACT})
