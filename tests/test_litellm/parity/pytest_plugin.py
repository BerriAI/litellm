from __future__ import annotations

import pytest

from tests.test_litellm.parity.compare import parity_comparison


def pytest_assertrepr_compare(config: pytest.Config, op: str, left: object, right: object) -> list[str] | None:
    if op != "==":
        return None
    return parity_comparison(left, right)
