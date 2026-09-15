from __future__ import annotations

import json
from collections.abc import Callable
from types import ModuleType
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils import token_counter as python_counter
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import token_counter as bridge
from litellm.rust_bridge.configuration import (
    CapabilityDefinition,
    ComponentName,
    ExecutionDecision,
    RolloutPolicy,
    RustImplementationState,
    _parse_env_bool,  # pyright: ignore[reportPrivateUsage]  # directly test env parsing contract
)
from litellm.rust_bridge.errors import RustRouteUnsupportedError
from litellm.rust_bridge.route import NativeComponent
from litellm.rust_bridge.token_counter import definition
from litellm.rust_bridge.token_counter import COMPONENT


@pytest.mark.parametrize(("value", "expected"), (("1", True), ("0", False), (" 1 ", True), (" 0 ", False)))
def test_parse_env_bool_accepts_binary_values(value: str, expected: bool) -> None:
    assert _parse_env_bool(value) is expected


def test_parse_env_bool_preserves_unset_value() -> None:
    assert _parse_env_bool(None) is None


@pytest.mark.parametrize("value", ("enabled", "true", "false", "yes", "no", "on", "off", ""))
def test_parse_env_bool_rejects_unknown_value(value: str) -> None:
    with pytest.raises(ValueError, match="must be '1' or '0'"):
        _parse_env_bool(value)


@pytest.mark.parametrize("environment", (None, "0", "1"))
@pytest.mark.parametrize("override", (None, False, True))
def test_public_token_counter_stays_python_only(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, override: bool | None
) -> None:
    def unexpected_native_load() -> ModuleType:
        raise AssertionError("public token counting must not load Rust")

    calls: Final[list[str]] = []

    def python_count(text: str) -> int:
        calls.append(text)
        return len(text)

    configuration.reset_rust_configuration()
    if environment is None:
        monkeypatch.delenv("LITELLM_RUST", raising=False)
    else:
        monkeypatch.setenv("LITELLM_RUST", environment)
    if override is not None:
        litellm.rust(override)
    monkeypatch.setattr(bindings, "get_native_bridge", unexpected_native_load)
    monkeypatch.setattr(python_counter, "_get_count_function", lambda model, custom_tokenizer: python_count)
    try:
        assert COMPONENT.resolve().decision is ExecutionDecision.PYTHON
        assert litellm.token_counter(model="gpt-4o", text="hello") == 5
        assert calls == ["hello"]
    finally:
        configuration.reset_rust_configuration()


@pytest.mark.parametrize("disabled", (False, True))
@pytest.mark.parametrize("through_compatibility_wrapper", (False, True))
def test_public_counter_enforces_catalog_decision(
    monkeypatch: pytest.MonkeyPatch, disabled: bool, through_compatibility_wrapper: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_token_counter", disabled)
    monkeypatch.setattr(
        definition,
        "COMPONENT",
        NativeComponent(
            name=ComponentName.TOKEN_COUNTER,
            capability=CapabilityDefinition(
                rust=RustImplementationState.UNIMPLEMENTED,
                python_available=False,
                rollout=RolloutPolicy.UNSUPPORTED,
            ),
            exports=(),
        ),
    )
    counter: Final = litellm.token_counter if through_compatibility_wrapper else python_counter.token_counter
    with pytest.raises(RustRouteUnsupportedError, match="token_counter"):
        counter(model="gpt-4o", text="hello")


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", (None, "0", "1"))
@pytest.mark.parametrize("override", (None, False, True))
async def test_budget_direct_import_bypasses_public_rollout(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, override: bool | None
) -> None:
    from litellm.proxy.spend_tracking.budget_reservation import count_request_input_tokens

    configuration.reset_rust_configuration()
    if environment is None:
        monkeypatch.delenv("LITELLM_RUST", raising=False)
    else:
        monkeypatch.setenv("LITELLM_RUST", environment)
    if override is not None:
        litellm.rust(override)
    model: Final = "gpt-4o"
    messages: Final = [{"role": "user", "content": "hello"}]
    body: Final = json.dumps({"model": model, "messages": messages}).encode()
    native_calls: Final[list[bytes]] = []

    async def counter(
        body: bytes,
        kind: str | None,
        encoding: str,
        disabled: bool,
        legacy_accounting: bool,
        resource_loader: Callable[[str], str],
    ) -> object:
        native_calls.append(body)
        return {"model": model, "input_tokens": 42}

    bridge.TOKEN_COUNTER.override(counter)
    try:
        budget: Final = await count_request_input_tokens(
            request_body=json.loads(body), route="/v1/messages", llm_router=None, raw_body=body
        )
        assert budget == {model: 42}
        assert native_calls == [body]
    finally:
        bridge.TOKEN_COUNTER.reset()
        configuration.reset_rust_configuration()
