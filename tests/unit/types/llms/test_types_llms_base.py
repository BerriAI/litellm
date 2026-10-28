import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Final

import pytest
from pydantic import ConfigDict, create_model

from litellm.types.llms.base import LiteLLMBaseModel

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


def test_deferred_first_use_build_leaves_caller_locals_snapshot_untouched() -> None:
    class DeferredProbe(LiteLLMBaseModel):
        model_config = ConfigDict(defer_build=True)
        value: int

    def build(model: type[DeferredProbe]) -> list[str]:
        local_vars: Final = locals()
        later: Final = 1
        _ = model(value=later)
        return sorted(local_vars)

    assert not DeferredProbe.__pydantic_complete__
    assert build(DeferredProbe) == ["model"]
    assert DeferredProbe.__pydantic_complete__


_RACE_ROUNDS: Final = 100
_RACE_THREADS: Final = 16
_RACE_VALIDATIONS_PER_THREAD: Final = 5
_RACE_FIELD_COUNT: Final = 20


class _Deferred(LiteLLMBaseModel):
    model_config = ConfigDict(defer_build=True)


def _fresh_deferred_subclass(round_id: int) -> tuple[type[LiteLLMBaseModel], type[LiteLLMBaseModel]]:
    fields: Final = {f"field_{index}": (str | int | None, None) for index in range(_RACE_FIELD_COUNT)}
    parent: Final = create_model(f"Parent{round_id}", __base__=_Deferred, **fields)
    child: Final = create_model(f"Child{round_id}", __base__=parent, extra_flag=(bool | None, False))
    return parent, child


def _first_use_outcome(child: type[LiteLLMBaseModel]) -> str:
    try:
        return type(child.model_validate({"field_0": "x"})).__name__
    except AttributeError as error:
        return f"{type(error).__name__}: {error}"


def _validate_after_barrier(child: type[LiteLLMBaseModel], gate: threading.Barrier) -> tuple[str, ...]:
    gate.wait()
    return tuple(_first_use_outcome(child) for _ in range(_RACE_VALIDATIONS_PER_THREAD))


def _concurrent_first_use_outcomes(child: type[LiteLLMBaseModel]) -> frozenset[str]:
    gate: Final = threading.Barrier(_RACE_THREADS)
    with ThreadPoolExecutor(max_workers=_RACE_THREADS) as executor:
        per_thread: Final = tuple(executor.map(lambda _: _validate_after_barrier(child, gate), range(_RACE_THREADS)))
    return frozenset(chain.from_iterable(per_thread))


def test_concurrent_first_use_of_a_deferred_subclass_always_builds_that_subclass() -> None:
    previous_switch_interval: Final = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for round_id in range(_RACE_ROUNDS):
            parent, child = _fresh_deferred_subclass(round_id)
            parent.model_validate({})
            assert not child.__pydantic_complete__
            assert _concurrent_first_use_outcomes(child) == {child.__name__}, f"round {round_id}"
    finally:
        sys.setswitchinterval(previous_switch_interval)
