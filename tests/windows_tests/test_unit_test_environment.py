import os
import subprocess
import sys

import pytest

from tests.unit.host_environment import is_host_only


@pytest.mark.skipif(os.name != "nt", reason="Winsock needs SYSTEMROOT; the POSIX allowlist is unaffected")
def test_child_python_starts_under_the_unit_test_environment_allowlist():
    environment = {name: value for name, value in os.environ.items() if not is_host_only(name)}
    child = subprocess.run(
        [sys.executable, "-c", "import asyncio; print('ok')"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "ok"
