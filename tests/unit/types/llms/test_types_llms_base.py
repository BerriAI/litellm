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


def _run_probe(probe: str, defer_pydantic_build: str | None) -> str:
    env: Final = {key: value for key, value in os.environ.items() if key != "DEFER_PYDANTIC_BUILD"}
    overrides: Final = {} if defer_pydantic_build is None else {"DEFER_PYDANTIC_BUILD": defer_pydantic_build}
    result: Final = subprocess.run(
        (sys.executable, "-I", "-c", probe),
        env={**env, **overrides},
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return result.stdout.strip().splitlines()[-1]


def _probe_with(defer_pydantic_build: str | None) -> str:
    return _run_probe(_PROBE, defer_pydantic_build)


@pytest.mark.parametrize("defer_pydantic_build", [None, "true", "1", "on"])
def test_litellm_models_defer_schema_build_until_first_use(defer_pydantic_build: str | None) -> None:
    assert _probe_with(defer_pydantic_build) == "False 1 True"


@pytest.mark.parametrize("defer_pydantic_build", ["false", "0", "off"])
def test_litellm_models_build_schema_at_class_creation_when_defer_is_disabled(defer_pydantic_build: str) -> None:
    assert _probe_with(defer_pydantic_build) == "True 1 True"


_NESTED_PROBE: Final = """
from pydantic import BaseModel

from litellm.types.llms.base import LiteLLMBaseModel


class Inner(LiteLLMBaseModel):
    value: int


class Parent(LiteLLMBaseModel):
    inner: Inner


class Holder(BaseModel):
    item: object


validated = Parent(inner={"value": 1}).inner
constructed = Inner.model_construct(value=2)
print(Holder(item=validated).model_dump_json(serialize_as_any=True), Holder(item=constructed).model_dump_json())
"""


def test_deferred_instances_created_without_their_own_init_still_serialize_as_any() -> None:
    assert _run_probe(_NESTED_PROBE, None) == '{"item":{"value":1}} {"item":{"value":2}}'
