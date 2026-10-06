import os
import subprocess
import sys
from typing import Final

import pytest

_PROBE: Final = """
from litellm.types.llms.base import LiteLLMBaseModel


class Probe(LiteLLMBaseModel):
    value: int


print(Probe.__pydantic_complete__, Probe(value=1).value, Probe.__pydantic_complete__)
"""


def _probe_with(defer_pydantic_build: str | None) -> str:
    env: Final = {key: value for key, value in os.environ.items() if key != "DEFER_PYDANTIC_BUILD"}
    overrides: Final = {} if defer_pydantic_build is None else {"DEFER_PYDANTIC_BUILD": defer_pydantic_build}
    result: Final = subprocess.run(
        (sys.executable, "-c", _PROBE),
        env={**env, **overrides},
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("defer_pydantic_build", [None, "true", "1", "on"])
def test_litellm_models_defer_schema_build_until_first_use(defer_pydantic_build: str | None) -> None:
    assert _probe_with(defer_pydantic_build) == "False 1 True"


@pytest.mark.parametrize("defer_pydantic_build", ["false", "0", "off"])
def test_litellm_models_build_schema_at_class_creation_when_defer_is_disabled(defer_pydantic_build: str) -> None:
    assert _probe_with(defer_pydantic_build) == "True 1 True"
