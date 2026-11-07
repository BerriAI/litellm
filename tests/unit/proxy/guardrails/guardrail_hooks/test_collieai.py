from __future__ import annotations

import importlib.util
from types import ModuleType, SimpleNamespace
from typing import Final

import httpx
import pytest
import respx
from fastapi import HTTPException

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_endpoints import (
    CreateGuardrailRequest,
    create_guardrail,
    get_guardrail_ui_settings,
    get_provider_specific_params,
)
from litellm.proxy.guardrails.guardrail_hooks.collieai import CollieAiGuardrail, initialize_guardrail
from litellm.proxy.guardrails.guardrail_hooks.collieai.collieai import (
    MISSING_PACKAGE_MESSAGE,
    CollieAiSetupError,
    PACKAGE_MODULE,
    has_readiness_guard,
    load_base,
    package_guardrail,
)
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import Guardrail, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.collieai import (
    DEFAULT_API_BASE,
    CollieAiGuardrailConfigModel,
    CollieAiGuardrailConfigModelOptionalParams,
)

PACKAGE_INSTALLED: Final = importlib.util.find_spec("collieai_guardrail") is not None
API_KEY: Final = "clai_" + "0" * 40
CHAT_REQUEST: Final = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "hello"}]}


class _RecordingGuardrail(CustomGuardrail):
    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        guardrail_name: str | None = None,
        event_hook: object = None,
        default_on: bool | None = None,
        **settings: object,
    ) -> None:
        super().__init__(guardrail_name=guardrail_name, default_on=bool(default_on))
        self.given = {
            "api_base": api_base,
            "api_key": api_key,
            "guardrail_name": guardrail_name,
            "event_hook": event_hook,
            "default_on": default_on,
            **settings,
        }


def _params(**extras: object) -> LitellmParams:
    return LitellmParams(
        guardrail="collieai", mode=["pre_call", "post_call"], default_on=True, api_key=API_KEY, **extras
    )


def _guardrail(litellm_params: LitellmParams) -> Guardrail:
    return Guardrail(guardrail_name="collieai", litellm_params=litellm_params)


def _init(litellm_params: LitellmParams) -> _RecordingGuardrail:
    callback: Final = initialize_guardrail(
        litellm_params, _guardrail(litellm_params), guardrail_cls=_RecordingGuardrail
    )
    assert isinstance(callback, _RecordingGuardrail)
    return callback


@pytest.fixture(autouse=True)
def _isolate_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])


def test_hands_the_entry_and_every_setting_to_the_package() -> None:
    callback: Final = _init(
        _params(
            api_base="https://collieai.internal.example/v1/integrations/litellm",
            enforcement_mode="block",
            deadline_ms=2000,
            network_margin_ms=300,
            max_body_bytes=1024,
            forward_conversation_id=True,
            summary_interval_s=30,
            synthetic_interval_s=0,
            workspace_id="not-a-collieai-setting",
        )
    )

    assert callback.given == {
        "api_base": "https://collieai.internal.example/v1/integrations/litellm",
        "api_key": API_KEY,
        "guardrail_name": "collieai",
        "event_hook": ["pre_call", "post_call"],
        "default_on": True,
        "enforcement_mode": "block",
        "deadline_ms": 2000,
        "network_margin_ms": 300,
        "max_body_bytes": 1024,
        "forward_conversation_id": True,
        "summary_interval_s": 30,
        "synthetic_interval_s": 0,
        "workspace_id": "not-a-collieai-setting",
    }
    assert litellm.callbacks == [callback]


def test_points_at_app_collieai_io_unless_told_otherwise() -> None:
    callback: Final = _init(_params())

    # The hosted service's base URL, from https://docs.collieai.io/integrations/litellm (2026-10-29)
    assert callback.given["api_base"] == DEFAULT_API_BASE == "https://app.collieai.io/v1/integrations/litellm"
    assert set(callback.given) == {"api_base", "api_key", "guardrail_name", "event_hook", "default_on"}


def test_hands_each_entry_its_own_name_phase_and_default_on() -> None:
    litellm_params: Final = LitellmParams(guardrail="collieai", mode="pre_call", default_on=False, api_key=API_KEY)
    callback: Final = initialize_guardrail(
        litellm_params,
        Guardrail(guardrail_name="collieai-prompts", litellm_params=litellm_params),
        guardrail_cls=_RecordingGuardrail,
    )

    assert isinstance(callback, _RecordingGuardrail)
    assert (callback.given["guardrail_name"], callback.given["event_hook"], callback.given["default_on"]) == (
        "collieai-prompts",
        "pre_call",
        False,
    )


def test_takes_the_key_from_collieai_api_key_only_when_the_entry_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from_environment: Final = "clai_" + "1" * 40
    monkeypatch.setenv("COLLIEAI_API_KEY", from_environment)
    without_key: Final = LitellmParams(guardrail="collieai", mode=["pre_call", "post_call"], default_on=True)

    assert _init(without_key).given["api_key"] == from_environment
    assert _init(_params()).given["api_key"] == API_KEY


def test_ui_form_defaults_are_what_the_package_gets() -> None:
    optional: Final = CollieAiGuardrailConfigModelOptionalParams()
    model: Final = CollieAiGuardrailConfigModel(api_key=API_KEY)
    callback: Final = _init(
        _params(**model.model_dump(exclude={"api_key", "optional_params"}), **optional.model_dump())
    )

    assert callback.given["api_base"] == model.api_base
    assert callback.given["deadline_ms"] == optional.deadline_ms
    assert callback.given["forward_conversation_id"] == optional.forward_conversation_id


@pytest.mark.asyncio
async def test_ui_offers_collieai_fields_without_the_package() -> None:
    assert CollieAiGuardrail.get_config_model() is CollieAiGuardrailConfigModel

    fields: Final = (await get_provider_specific_params())["collieai"]

    assert fields["ui_friendly_name"] == "CollieAi"
    assert fields["api_key"]["required"] is True
    assert fields["api_base"]["default_value"] == DEFAULT_API_BASE
    assert set(fields["optional_params"]["fields"]) == {"deadline_ms", "forward_conversation_id"}


@pytest.mark.asyncio
async def test_ui_offers_only_the_two_phases_collieai_checks() -> None:
    settings: Final = await get_guardrail_ui_settings()

    assert settings.supported_modes_by_provider["collieai"] == ["pre_call", "post_call"]


def test_a_package_without_the_readiness_guard_is_too_old() -> None:
    assert not has_readiness_guard(ModuleType("collieai_guardrail.guardrail"))
    assert has_readiness_guard(SimpleNamespace(require_guardrail=object()))


def test_only_a_current_package_s_guardrail_class_is_taken() -> None:
    guard: Final = object()

    assert package_guardrail(SimpleNamespace(require_guardrail=guard, CollieAIGuardrail=_RecordingGuardrail)) is (
        _RecordingGuardrail
    )
    assert package_guardrail(SimpleNamespace(CollieAIGuardrail=_RecordingGuardrail)) is None
    assert package_guardrail(SimpleNamespace(require_guardrail=guard, CollieAIGuardrail=object())) is None
    assert package_guardrail(SimpleNamespace(require_guardrail=guard, CollieAIGuardrail=ValueError)) is None


def test_builds_on_the_package_only_when_it_is_current() -> None:
    current: Final = SimpleNamespace(require_guardrail=object(), CollieAIGuardrail=_RecordingGuardrail)
    too_old: Final = SimpleNamespace(CollieAIGuardrail=_RecordingGuardrail)

    assert load_base({PACKAGE_MODULE: current}.__getitem__) is _RecordingGuardrail
    with pytest.raises(ImportError, match="pip install") as raised:
        load_base({PACKAGE_MODULE: too_old}.__getitem__)(guardrail_name="collieai")

    assert str(raised.value) == MISSING_PACKAGE_MESSAGE
    assert raised.value.__cause__ is None


def test_without_the_package_building_the_guardrail_names_what_to_install() -> None:
    cause: Final = ImportError("No module named 'collieai_guardrail'")

    def import_module(name: str) -> object:
        raise cause

    with pytest.raises(ImportError, match="pip install") as raised:
        load_base(import_module)(guardrail_name="collieai")

    assert str(raised.value) == MISSING_PACKAGE_MESSAGE
    assert raised.value.__cause__ is cause


@pytest.mark.skipif(PACKAGE_INSTALLED, reason="exercises the missing-package fallback")
def test_missing_package_fails_at_config_load_with_install_hint() -> None:
    with pytest.raises(CollieAiSetupError, match="pip install") as raised:
        InMemoryGuardrailHandler().initialize_guardrail(_guardrail(_params()))

    assert not isinstance(raised.value, (ValueError, TypeError))
    assert litellm.callbacks == []


@pytest.mark.asyncio
async def test_creating_a_guardrail_the_package_refuses_answers_400_and_removes_the_row(mocker) -> None:
    from unittest.mock import AsyncMock

    litellm_params: Final = {
        "guardrail": "collieai",
        "mode": ["pre_call", "post_call"],
        "default_on": True,
        "api_key": "sk-not-a-collieai-key",
    }
    row: Final = {"guardrail_name": "collieai-refused", "litellm_params": litellm_params, "guardrail_info": {}}
    registry: Final = mocker.Mock(add_guardrail_to_db=AsyncMock(return_value={**row, "guardrail_id": "refused"}))
    mocker.patch("litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY", registry)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mocker.Mock())
    mocker.patch("litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER", InMemoryGuardrailHandler())
    delete_row: Final = mocker.patch("litellm.proxy.guardrails.guardrail_endpoints._delete_guardrail_row", AsyncMock())

    with pytest.raises(HTTPException) as refused:
        await create_guardrail(
            CreateGuardrailRequest(guardrail=row),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert refused.value.status_code == 400
    delete_row.assert_awaited_once()
    assert litellm.callbacks == []


def _load(monkeypatch: pytest.MonkeyPatch, **extras: object) -> CustomGuardrail:
    """The guardrail as the proxy loads `guardrail: collieai` from its config, with no synthetic check at start. Block
    mode also needs the package's readiness guard among the proxy's callbacks, which the config file loads first"""
    from collieai_guardrail.guardrail import require_guardrail

    block: Final = extras.get("enforcement_mode") == "block"
    monkeypatch.setattr(litellm, "callbacks", [require_guardrail] if block else [])
    InMemoryGuardrailHandler().initialize_guardrail(_guardrail(_params(synthetic_interval_s=0, **extras)))
    loaded: Final = [callback for callback in litellm.callbacks if isinstance(callback, CollieAiGuardrail)]
    assert len(loaded) == 1
    return loaded[0]


async def _pre_call() -> dict[str, object]:
    data: Final = {**CHAT_REQUEST, "metadata": {}}
    await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"), data=data, call_type="acompletion"
    )
    return data


@pytest.mark.skipif(not PACKAGE_INSTALLED, reason="needs collieai-litellm-guardrail")
@pytest.mark.asyncio
@respx.mock
async def test_block_refuses_the_call_when_collieai_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    route: Final = respx.post(url__startswith=DEFAULT_API_BASE).mock(side_effect=httpx.ConnectError("refused"))
    _load(monkeypatch, enforcement_mode="block")

    with pytest.raises((GuardrailRaisedException, HTTPException)) as refused:
        await _pre_call()

    assert route.called
    assert getattr(refused.value, "status_code", None) == 503


@pytest.mark.skipif(not PACKAGE_INSTALLED, reason="needs collieai-litellm-guardrail")
@pytest.mark.asyncio
@respx.mock
async def test_monitor_lets_the_call_go_on_and_records_why(monkeypatch: pytest.MonkeyPatch) -> None:
    route: Final = respx.post(url__startswith=DEFAULT_API_BASE).mock(side_effect=httpx.ConnectError("refused"))
    _load(monkeypatch)

    data: Final = await _pre_call()

    assert route.called
    metadata: Final = data["metadata"]
    assert isinstance(metadata, dict)
    [record] = metadata["standard_logging_guardrail_information"]
    assert (record["guardrail_provider"], record["guardrail_status"]) == ("collieai", "guardrail_failed_to_respond")
