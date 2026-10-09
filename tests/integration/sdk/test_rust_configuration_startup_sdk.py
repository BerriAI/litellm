import os
from typing import Final

import pytest

from tests._support.child_interpreter import run_child_interpreter


@pytest.mark.parametrize(("value", "expected"), (("1", "True"), ("0", "False")))
def test_environment_controls_startup(value: str, expected: str) -> None:
    result: Final = run_child_interpreter(
        "from litellm.rust_bridge.configuration import rust_enabled; print(rust_enabled())",
        env={**os.environ, "LITELLM_RUST": value},
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
