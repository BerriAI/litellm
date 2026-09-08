import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("configured, expected", [("false", "False"), ("true", "True")])
def test_litellm_drop_params_env_var_is_parsed_as_a_flag(configured, expected):
    result = subprocess.run(
        [sys.executable, "-c", "import litellm; print(litellm.drop_params)"],
        env={**os.environ, "LITELLM_DROP_PARAMS": configured},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected
