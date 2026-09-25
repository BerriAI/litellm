import asyncio
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import _PROXY_DynamicRateLimitHandlerV3
from litellm.proxy.hooks.fairness_settings import apply_fairness_settings
from litellm.proxy.management_endpoints.fairness_endpoints import (
    get_fairness_settings,
    get_fairness_status,
    update_fairness_settings,
)
from litellm.router import Router
from litellm.types.proxy.fairness import FairnessSettings, WorkloadClass


def _auth(role: LitellmUserRoles) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="hashed", user_id="user", user_role=role)


class _FakeProxyConfig:
    def __init__(self, litellm_settings: dict[str, object]) -> None:
        self.config: dict[str, dict[str, object]] = {"litellm_settings": dict(litellm_settings)}
        self.saved: list[dict[str, dict[str, object]]] = []

    async def get_config(self) -> dict[str, dict[str, object]]:
        return self.config

    async def save_config(self, new_config: dict[str, dict[str, object]]) -> None:
        self.saved.append(new_config)
        self.config = new_config


def _settings() -> FairnessSettings:
    return FairnessSettings(
        enabled=True,
        workload_classes=(
            WorkloadClass(name="production", reserved_share=0.6, max_queue_wait_seconds=30.0),
            WorkloadClass(name="batch", reserved_share=0.1, max_queue_wait_seconds=120.0),
        ),
        default_reserved_share=0.1,
        saturation_threshold=0.8,
    )


@pytest.fixture
def isolated_globals(monkeypatch: pytest.MonkeyPatch) -> _FakeProxyConfig:
    proxy_config: Final = _FakeProxyConfig({"drop_params": True})
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_config)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "fairness_settings", None)
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm, "priority_reservation_settings", None)
    return proxy_config


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM])
async def test_reads_reject_non_admin_roles(isolated_globals: _FakeProxyConfig, role: LitellmUserRoles) -> None:
    with pytest.raises(HTTPException) as settings_error:
        await get_fairness_settings(user_api_key_dict=_auth(role))
    with pytest.raises(HTTPException) as status_error:
        await get_fairness_status(user_api_key_dict=_auth(role))
    assert settings_error.value.status_code == 403
    assert status_error.value.status_code == 403


@pytest.mark.asyncio
async def test_reads_allow_proxy_admin_viewer(isolated_globals: _FakeProxyConfig) -> None:
    viewer: Final = _auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    settings: Final = await get_fairness_settings(user_api_key_dict=viewer)
    status: Final = await get_fairness_status(user_api_key_dict=viewer)
    assert settings.settings == FairnessSettings()
    assert settings.persisted is False
    assert status.enabled is False
    assert status.limiter_active is False
    assert status.models == ()


@pytest.mark.asyncio
async def test_update_rejects_proxy_admin_viewer(isolated_globals: _FakeProxyConfig) -> None:
    with pytest.raises(HTTPException) as error:
        await update_fairness_settings(
            settings=_settings(),
            user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
        )
    assert error.value.status_code == 403
    assert isolated_globals.saved == []
    assert litellm.fairness_settings is None


@pytest.mark.asyncio
async def test_update_requires_store_model_in_db(
    isolated_globals: _FakeProxyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(proxy_server, "store_model_in_db", False)
    with pytest.raises(HTTPException) as error:
        await update_fairness_settings(settings=_settings(), user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN))
    assert error.value.status_code == 500
    assert isolated_globals.saved == []


@pytest.mark.asyncio
async def test_update_persists_applies_and_audits(
    isolated_globals: _FakeProxyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_calls: list[dict[str, object]] = []

    async def capture_audit(**kwargs: object) -> None:
        audit_calls.append(kwargs)

    monkeypatch.setattr(proxy_server, "create_config_audit_log", capture_audit)
    monkeypatch.setattr(litellm, "callbacks", [])
    settings: Final = _settings()

    response: Final = await update_fairness_settings(
        settings=settings, user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN)
    )
    for _ in range(3):
        await asyncio.sleep(0)

    assert response.settings == settings
    assert response.persisted is True

    persisted_settings: Final = isolated_globals.config["litellm_settings"]
    assert persisted_settings["drop_params"] is True
    assert FairnessSettings.model_validate(persisted_settings["fairness_settings"]) == settings

    assert litellm.fairness_settings == settings
    assert litellm.priority_reservation == {"production": 0.6, "batch": 0.1}
    assert litellm.priority_reservation_settings is not None
    assert litellm.priority_reservation_settings.default_priority == 0.1
    assert litellm.priority_reservation_settings.saturation_threshold == 0.8
    assert any(
        callback == "dynamic_rate_limiter_v3" or isinstance(callback, _PROXY_DynamicRateLimitHandlerV3)
        for callback in litellm.callbacks
    )

    assert len(audit_calls) == 1
    assert audit_calls[0]["param_name"] == "fairness_settings"
    assert audit_calls[0]["action"] == "updated"
    assert audit_calls[0]["before_value"] is None
    assert audit_calls[0]["after_value"] == settings.model_dump(mode="json")

    read_back: Final = await get_fairness_settings(user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
    assert read_back.settings == settings
    assert read_back.persisted is True


@pytest.mark.asyncio
async def test_disabling_after_enabling_clears_mirrored_reservations_and_audits_previous_value(
    isolated_globals: _FakeProxyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_calls: list[dict[str, object]] = []

    async def capture_audit(**kwargs: object) -> None:
        audit_calls.append(kwargs)

    monkeypatch.setattr(proxy_server, "create_config_audit_log", capture_audit)
    monkeypatch.setattr(litellm, "callbacks", [])
    admin: Final = _auth(LitellmUserRoles.PROXY_ADMIN)
    first: Final = _settings()
    second: Final = first.model_copy(update={"enabled": False})

    await update_fairness_settings(settings=first, user_api_key_dict=admin)
    assert litellm.priority_reservation == {"production": 0.6, "batch": 0.1}
    await update_fairness_settings(settings=second, user_api_key_dict=admin)
    for _ in range(3):
        await asyncio.sleep(0)

    assert [call["before_value"] for call in audit_calls] == [None, first.model_dump(mode="json")]
    assert litellm.fairness_settings == second
    assert litellm.priority_reservation is None
    assert litellm.priority_reservation_settings is not None
    assert litellm.priority_reservation_settings.saturation_threshold != first.saturation_threshold
    status: Final = await get_fairness_status(user_api_key_dict=admin)
    assert status.enabled is False


def test_disabled_settings_leave_config_file_reservations_alone(
    isolated_globals: _FakeProxyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "priority_reservation", {"legacy": 0.5})
    apply_fairness_settings(FairnessSettings(enabled=False), internal_usage_cache=None, llm_router=None)
    assert litellm.priority_reservation == {"legacy": 0.5}


def test_settings_reject_duplicate_and_reserved_class_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        FairnessSettings(
            workload_classes=(
                WorkloadClass(name="batch", reserved_share=0.1),
                WorkloadClass(name="batch", reserved_share=0.2),
            )
        )
    with pytest.raises(ValidationError, match="reserved"):
        FairnessSettings(workload_classes=(WorkloadClass(name="default", reserved_share=0.1),))


def test_settings_reject_shares_that_exceed_full_capacity_including_default_pool() -> None:
    classes: Final = (
        WorkloadClass(name="production", reserved_share=0.6),
        WorkloadClass(name="batch", reserved_share=0.3),
    )
    with pytest.raises(ValidationError, match="add up to 115%"):
        FairnessSettings(workload_classes=classes, default_reserved_share=0.25)
    exact: Final = FairnessSettings(workload_classes=classes, default_reserved_share=0.1)
    assert sum(exact.reserved_shares().values()) + exact.default_reserved_share == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_status_reports_reserved_capacity_per_class(
    isolated_globals: _FakeProxyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "fair-model",
                "litellm_params": {"model": "openai/gpt-5.4", "api_key": "sk-test", "rpm": 100, "tpm": 10_000},
            }
        ]
    )
    limiter: Final = _PROXY_DynamicRateLimitHandlerV3(internal_usage_cache=DualCache())
    limiter.update_variables(llm_router=router)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "create_config_audit_log", _noop_audit)
    monkeypatch.setattr(litellm, "callbacks", [limiter])
    monkeypatch.setattr(litellm.litellm_core_utils.litellm_logging, "_in_memory_loggers", [limiter])

    await update_fairness_settings(settings=_settings(), user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN))
    status: Final = await get_fairness_status(user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN))

    assert status.enabled is True
    assert status.limiter_active is True
    assert status.saturation_threshold == 0.8
    assert [model.model_group for model in status.models] == ["fair-model"]
    model_status: Final = status.models[0]
    assert model_status.rpm == 100
    assert model_status.tpm == 10_000
    by_class: Final = {row.name: row for row in model_status.classes}
    assert by_class["production"].reserved_rpm == 60
    assert by_class["production"].reserved_tpm == 6_000
    assert by_class["production"].max_queue_wait_seconds == 30.0
    assert by_class["batch"].reserved_rpm == 10
    assert by_class["batch"].max_queue_wait_seconds == 120.0
    assert by_class["default"].reserved_rpm == 10


async def _noop_audit(**kwargs: object) -> None:
    return None
