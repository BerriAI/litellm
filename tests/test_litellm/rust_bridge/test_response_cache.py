from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import CacheFacadeRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.response_cache import select_cache_facade


class PythonFacade:
    pass


class NativeFacade:
    pass


def native_facade(value: type | None) -> NativeBinding[type]:
    binding: Final[NativeBinding[type]] = NativeBinding("unused", validate=lambda _: None)
    binding.override(value)
    return binding


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()


@pytest.mark.parametrize("rules", ((CacheFacadeRule(Rollout.PYTHON_ONLY),), ()))
def test_python_rules_keep_the_python_facade_even_when_rust_is_enabled(rules: Rules) -> None:
    configuration.rust(True)
    assert select_cache_facade(PythonFacade, rules, native_facade(NativeFacade)) is PythonFacade


@pytest.mark.parametrize(
    "rollout",
    (Rollout.RUST_REQUIRED, Rollout.RUST_OPT_OUT),
)
def test_rust_rules_select_the_native_facade(rollout: Rollout) -> None:
    rules: Final = (CacheFacadeRule(rollout),)
    assert select_cache_facade(PythonFacade, rules, native_facade(NativeFacade)) is NativeFacade


def test_opt_in_rule_follows_the_global_switch() -> None:
    rules: Final = (CacheFacadeRule(Rollout.RUST_OPT_IN),)
    assert select_cache_facade(PythonFacade, rules, native_facade(NativeFacade)) is PythonFacade
    configuration.rust(True)
    assert select_cache_facade(PythonFacade, rules, native_facade(NativeFacade)) is NativeFacade


def test_missing_native_facade_falls_back_unless_required() -> None:
    fallback: Final = (CacheFacadeRule(Rollout.RUST_OPT_OUT),)
    assert select_cache_facade(PythonFacade, fallback, native_facade(None)) is PythonFacade
    with pytest.raises(RuntimeError, match="Rust cache facade is unavailable"):
        select_cache_facade(PythonFacade, (CacheFacadeRule(Rollout.RUST_REQUIRED),), native_facade(None))
