"""Every module in this package that defines a `CASE` is one perf test."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Final

from setup.performance_test_case import PerformanceTestCase


def _load() -> tuple[PerformanceTestCase, ...]:
    found: list[PerformanceTestCase] = []
    for module_info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        case: object = getattr(module, "CASE", None)
        if isinstance(case, PerformanceTestCase):
            found.append(case)
    return tuple(sorted(found, key=lambda case: case.name))


CASES: Final = _load()
