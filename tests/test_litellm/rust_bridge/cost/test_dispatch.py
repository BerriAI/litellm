from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest
from typing_extensions import Never

from litellm import cost_calculator
from litellm.litellm_core_utils.llm_cost_calc import guardrail_cost, utils
from litellm.rust_bridge import catalog
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import CostRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.cost.dispatch import NATIVE_COST_API, CostApi, run_cost_api


class NativeCalled(Exception):
    pass


def _native_binding(native: Callable[[str], Never] | None) -> NativeBinding[Callable[[str], Never]]:
    binding: Final[NativeBinding[Callable[[str], Never]]] = NativeBinding("cost_api", validate=lambda value: None)
    binding.override(native)
    return binding


def test_python_only_runs_cost_api_without_loading_native() -> None:
    def native(name: str) -> Never:
        raise NativeCalled(name)

    assert run_cost_api(CostApi.COST_PER_TOKEN, lambda: 42, binding=_native_binding(native)) == 42


def test_rust_required_sends_api_to_native_without_python_fallback() -> None:
    def native(name: str) -> Never:
        raise NativeCalled(name)

    rules: Final[Rules] = (CostRule(Rollout.RUST_REQUIRED),)
    with pytest.raises(NativeCalled, match=r"^cost_per_token$"):
        run_cost_api(
            CostApi.COST_PER_TOKEN,
            lambda: pytest.fail("Python cost implementation ran"),
            rules=rules,
            binding=_native_binding(native),
        )


def test_rust_required_fails_when_native_is_unavailable() -> None:
    rules: Final[Rules] = (CostRule(Rollout.RUST_REQUIRED),)

    with pytest.raises(RuntimeError, match="Rust cost API cost_per_token is unavailable"):
        run_cost_api(
            CostApi.COST_PER_TOKEN,
            lambda: pytest.fail("Python cost implementation ran"),
            rules=rules,
            binding=_native_binding(None),
        )


@pytest.mark.parametrize("api", tuple(CostApi))
def test_rust_required_routes_each_public_cost_function(api: CostApi, monkeypatch: pytest.MonkeyPatch) -> None:
    def native(name: str) -> Never:
        raise NativeCalled(name)

    owner: Final = next(
        owner
        for owner in (cost_calculator, utils, utils.CostCalculatorUtils, guardrail_cost)
        if hasattr(owner, api.value)
    )
    target: Final[object] = getattr(owner, api.value)  # pyright: ignore[reportAny]  # module lookup is dynamic
    assert callable(target)
    monkeypatch.setattr(catalog, "RULES", (CostRule(Rollout.RUST_REQUIRED),))
    NATIVE_COST_API.override(native)
    try:
        with pytest.raises(NativeCalled, match=rf"^{api.value}$"):
            target()
    finally:
        NATIVE_COST_API.reset()
