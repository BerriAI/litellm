from collections.abc import Sequence
from typing import Final

import pytest

from litellm.proxy.management_endpoints.liteask.approval import ScriptRunner


class AtomicApprovalStore:
    def __init__(self, *, issue_result: object = 1, unavailable: bool = False) -> None:
        self.records: dict[str, str | bytes | int | float] = {}
        self.issue_result: Final = issue_result
        self.unavailable: Final = unavailable

    def lose_state(self) -> None:
        self.records.clear()

    def async_register_script(self, script: str) -> ScriptRunner:
        async def execute(keys: Sequence[str], args: Sequence[str | bytes | int | float]) -> object:
            if self.unavailable:
                raise ConnectionError("offline")
            assert len(keys) == 1 and keys[0].startswith("liteask:approval:")
            if "'SET'" in script:
                assert len(args) == 2 and isinstance(args[1], int) and args[1] > 0
                if keys[0] in self.records:
                    return 0
                self.records[keys[0]] = args[0]
                return self.issue_result
            assert "'DEL'" in script and len(args) == 1
            if self.records.get(keys[0]) != args[0]:
                return 0
            del self.records[keys[0]]
            return 1

        return execute


@pytest.fixture
def approval_store() -> AtomicApprovalStore:
    return AtomicApprovalStore()
