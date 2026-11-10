from typing import Final

import openapi_docs_gate as gate


def test_base_counts_imports_native_extension_from_detached_worktree() -> None:
    expected_counts: Final = dict(gate.count_by_rule(gate.head_violations()))
    assert expected_counts
    assert dict(gate.base_counts("HEAD")) == expected_counts
