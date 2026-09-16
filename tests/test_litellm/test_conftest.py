import os
import subprocess
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]

PROXY_BASE_URL_SENSITIVE_NODE: Final = (
    "tests/test_litellm/proxy/management_endpoints/test_mcp_management_endpoints.py"
    "::TestTemporaryMCPSessionEndpoints"
    "::test_mcp_token_opens_sealed_passthrough_code_and_exchanges_with_minted_client"
)

COVERAGE_SUBPROCESS_VARS: Final = frozenset(
    {"COV_CORE_SOURCE", "COV_CORE_CONFIG", "COV_CORE_DATAFILE", "COV_CORE_CONTEXT", "COVERAGE_PROCESS_START"}
)


def test_host_proxy_base_url_cannot_reach_request_derived_url_tests():
    child_env: Final = {
        key: value for key, value in os.environ.items() if key not in COVERAGE_SUBPROCESS_VARS
    } | {"PROXY_BASE_URL": "https://leaked-host-origin.example.com"}

    completed: Final = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            PROXY_BASE_URL_SENSITIVE_NODE,
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
