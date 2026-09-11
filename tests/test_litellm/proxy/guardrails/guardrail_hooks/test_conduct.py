from __future__ import annotations

import importlib.util
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_hooks.conduct import (
    DEFAULT_TIMEOUT_SECONDS,
    ConductGuardrail,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.types.guardrails import Guardrail, GuardrailEventHooks, LitellmParams

PACKAGE_INSTALLED: Final = importlib.util.find_spec("conduct_litellm_guard") is not None


class _RecordingGuardrail(CustomGuardrail):
    """Stand-in with the ``conduct_litellm_guard.ConductGuard`` class contract."""

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [GuardrailEventHooks.pre_call]

    def __init__(
        self,
        *,
        api_url: str | None = None,
        agent_token: str | None = None,
        workspace_id: str | None = None,
        fail_mode: str = "fail_closed",
        tool_name: str = "llm_call",
        timeout: float = 8.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # pyright: ignore[reportArgumentType]  # CustomGuardrail.__init__ is untyped
        self.api_url = api_url
        self.agent_token = agent_token
        self.workspace_id = workspace_id
        self.fail_mode = fail_mode
        self.tool_name = tool_name
        self.timeout = timeout


def _params(mode: str = "pre_call", **extras: object) -> LitellmParams:
    return LitellmParams(guardrail="conduct", mode=mode, api_key="cond_agt_test", **extras)


def _guardrail(litellm_params: LitellmParams) -> Guardrail:
    return Guardrail(guardrail_name="conduct-guard", litellm_params=litellm_params)


def _init(litellm_params: LitellmParams) -> _RecordingGuardrail:
    callback: Final = initialize_guardrail(
        litellm_params, _guardrail(litellm_params), guardrail_cls=_RecordingGuardrail
    )
    assert isinstance(callback, _RecordingGuardrail)
    return callback


@pytest.fixture(autouse=True)
def _isolate_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])


def test_discovered_by_guardrail_registry() -> None:
    assert guardrail_initializer_registry["conduct"] is initialize_guardrail
    assert guardrail_class_registry["conduct"] is ConductGuardrail


def test_maps_typed_fields_and_extras_onto_plugin_kwargs() -> None:
    callback: Final = _init(
        _params(
            api_base="https://guard.example.test",
            unreachable_fallback="fail_open",
            timeout="3",
            workspace_id="ws_123",
            tool_name="workflow",
            default_on=True,
        )
    )

    assert callback.api_url == "https://guard.example.test"
    assert callback.agent_token == "cond_agt_test"
    assert callback.fail_mode == "fail_open"
    assert callback.timeout == 3.0
    assert callback.workspace_id == "ws_123"
    assert callback.tool_name == "workflow"
    assert callback.guardrail_name == "conduct-guard"
    assert callback.event_hook == "pre_call"
    assert callback.default_on is True
    assert litellm.callbacks == [callback]


def test_defaults_when_optional_config_is_omitted() -> None:
    callback: Final = _init(_params())

    assert callback.fail_mode == "fail_closed"
    assert callback.timeout == DEFAULT_TIMEOUT_SECONDS
    assert callback.workspace_id is None
    assert callback.tool_name == "llm_call"


@pytest.mark.parametrize("mode", ["during_call", "post_call", "logging_only"])
def test_rejects_modes_the_plugin_does_not_implement(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_STRICT_GUARDRAIL_MODES", raising=False)

    with pytest.raises(ValueError, match="not in the supported event hooks"):
        _init(_params(mode=mode))

    assert litellm.callbacks == []


@pytest.mark.skipif(PACKAGE_INSTALLED, reason="exercises the missing-package fallback")
def test_missing_package_fails_at_config_load_with_install_hint() -> None:
    litellm_params: Final = _params()

    with pytest.raises(ImportError, match="pip install"):
        initialize_guardrail(litellm_params, _guardrail(litellm_params))

    assert litellm.callbacks == []


@pytest.mark.skipif(not PACKAGE_INSTALLED, reason="needs conduct-litellm-guard")
def test_plugin_class_enforces_supported_modes() -> None:
    assert ConductGuardrail.get_supported_event_hooks() == [GuardrailEventHooks.pre_call]

    accepted: Final = _params()
    callback: Final = initialize_guardrail(accepted, _guardrail(accepted))
    assert callback.supported_event_hooks == [GuardrailEventHooks.pre_call]
    assert litellm.callbacks == [callback]

    rejected: Final = _params(mode="during_call")
    with pytest.raises(ValueError, match="not in the supported event hooks"):
        initialize_guardrail(rejected, _guardrail(rejected))
