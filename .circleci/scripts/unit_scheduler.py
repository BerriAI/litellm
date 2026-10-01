from typing import Final

import pytest
from xdist.remote import Producer
from xdist.scheduler import LoadScopeScheduling

_INDEPENDENT_CASES: Final = (
    "tests/unit/experimental_mcp_client/test_mcp_client.py::test_cancellation_delivers_termination_over_tcp[",
    "tests/unit/enterprise/enterprise_callbacks/test_secret_detection.py::test_scan_message_keeps_benign_values[",
    "tests/unit/enterprise/enterprise_callbacks/test_secret_detection.py"
    "::test_scan_message_redacts_credentials_assigned_to_credential_keys[",
)


class UnitScheduling(LoadScopeScheduling):
    def _split_scope(self, nodeid: str) -> str:
        if nodeid.startswith(_INDEPENDENT_CASES):
            return nodeid
        return super()._split_scope(nodeid)


def pytest_xdist_make_scheduler(config: pytest.Config, log: Producer) -> LoadScopeScheduling | None:
    if config.getoption("dist") == "loadscope":
        return UnitScheduling(config, log)
    return None
