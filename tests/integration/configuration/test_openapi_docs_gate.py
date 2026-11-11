from typing import Final

import openapi_docs_gate as gate
import pytest


def test_base_counts_imports_native_extension_from_detached_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_OFFLINE", "1")
    expected_counts: Final = dict(gate.count_by_rule(gate.head_violations()))
    assert expected_counts
    assert dict(gate.base_counts("HEAD")) == expected_counts
