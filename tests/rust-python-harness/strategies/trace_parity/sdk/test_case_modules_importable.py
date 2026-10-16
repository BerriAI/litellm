from __future__ import annotations

from importlib import import_module
from typing import Final, cast

import pytest

from ..models import TraceSuite


@pytest.mark.parametrize(
    "module",
    [
        "tests.rust-python-harness.strategies.trace_parity.sdk.messages.case",
        "tests.rust-python-harness.strategies.trace_parity.sdk.chat_completions.case",
        "tests.rust-python-harness.strategies.trace_parity.sdk.transcription.case",
    ],
)
def test_implemented_namespace_case_modules_remain_importable(module: str) -> None:
    loaded: Final = import_module(module)
    suite: Final = cast(object, getattr(loaded, "TRACE_SUITE"))
    assert isinstance(suite, TraceSuite)
    assert suite.scenarios
