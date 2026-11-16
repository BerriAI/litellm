import os
import subprocess
import sys
from typing import Final


def test_importing_proxy_hooks_does_not_mutate_litellm_callbacks():
    script: Final = (
        "import litellm; "
        "litellm.callbacks = []; "
        "import litellm.proxy.hooks; "
        "assert len(litellm.callbacks) == 0, f'mutated: {litellm.callbacks}'"
    )
    result: Final = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
