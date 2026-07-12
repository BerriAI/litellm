"""Unit coverage for the lifecycle harness (lifecycle.run_case).

Cases register cleanups progressively during init() (create team, then user, then
key), so a failure partway through init() must still release whatever was already
created on the long-lived shared proxy. This guards that contract.
"""

from dataclasses import dataclass, field
from typing import Callable, List

import pytest

from lifecycle import run_case


@dataclass
class _PartialInitCase:
    """init() registers a cleanup, then raises before finishing - mirroring a real
    case that creates a resource, registers its delete, then fails on the next
    step."""

    released: List[str] = field(default_factory=list)
    _undo: List[Callable[[], None]] = field(default_factory=list)

    def init(self) -> None:
        self._undo.append(lambda: self.released.append("first"))
        raise RuntimeError("init failed after registering the first resource")

    def run(self) -> None:
        raise AssertionError("run() must not execute when init() failed")

    def teardown(self) -> None:
        for undo in reversed(self._undo):
            undo()


def test_run_case_releases_resources_when_init_fails_partway() -> None:
    case = _PartialInitCase()

    with pytest.raises(RuntimeError, match="init failed"):
        run_case(case)

    assert case.released == ["first"], (
        "a resource registered before init() failed must still be released, or it "
        "leaks on the long-lived shared proxy"
    )
