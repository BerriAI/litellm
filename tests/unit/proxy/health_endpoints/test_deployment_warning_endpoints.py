import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import JsonValue

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import AUDIT_ACTIONS, LitellmTableNames, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.health_endpoints.deployment_warning_endpoints import (
    WarningCondition,
    config_audit_logger,
    current_warning_conditions,
    deployment_warning_conditions,
    deployment_warning_ignore_store,
    license_expiry_condition,
    router,
    utc_now,
)
from litellm.proxy.proxy_server import (
    _dump_redacted_config,  # pyright: ignore[reportPrivateUsage]  # the exact redaction every config audit row goes through
)
from litellm.repositories.deployment_warning_ignore_repository import DEPLOYMENT_WARNING_IGNORES_ROW_ID

T0: Final = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
THIRTY_DAYS: Final = 30 * 24 * 60 * 60


@dataclass
class InMemoryIgnoreStore:
    raw: object = None

    async def get_raw_ignores(self) -> object:
        return self.raw

    async def upsert_ignores(self, payload: str) -> None:
        self.raw = json.loads(payload)


@dataclass
class RecordingAuditLogger:
    calls: list[dict[str, object]] = field(default_factory=list)

    async def __call__(
        self,
        param_name: str,
        action: AUDIT_ACTIONS,
        before_value: JsonValue | None,
        after_value: JsonValue | None,
        user_api_key_dict: UserAPIKeyAuth,
        table_name: LitellmTableNames = LitellmTableNames.CONFIG_TABLE_NAME,
    ) -> None:
        self.calls.append(
            {
                "param_name": param_name,
                "action": action,
                "before_value": before_value,
                "after_value": after_value,
                "table_name": table_name,
            }
        )


@dataclass
class Harness:
    app: FastAPI
    client: TestClient
    store: InMemoryIgnoreStore
    audit: RecordingAuditLogger
    clock: list[datetime]

    def as_role(self, role: LitellmUserRoles, user_id: str | None = "admin-user") -> None:
        async def auth() -> UserAPIKeyAuth:
            return UserAPIKeyAuth(api_key="sk-test", user_id=user_id, user_role=role)

        self.app.dependency_overrides[user_api_key_auth] = auth

    def with_conditions(self, *conditions: WarningCondition) -> None:
        self.app.dependency_overrides[current_warning_conditions] = lambda: conditions

    def without_database(self) -> None:
        self.app.dependency_overrides[deployment_warning_ignore_store] = lambda: None

    def warnings(self) -> dict[str, dict[str, object]]:
        response = self.client.get("/health/deployment_warnings")
        assert response.status_code == 200, response.text
        return {warning["id"]: warning for warning in response.json()["warnings"]}


@pytest.fixture
def harness() -> Iterator[Harness]:
    app = FastAPI()
    app.include_router(router)
    store = InMemoryIgnoreStore()
    audit = RecordingAuditLogger()
    clock = [T0]
    app.dependency_overrides[deployment_warning_ignore_store] = lambda: store
    app.dependency_overrides[config_audit_logger] = lambda: audit
    app.dependency_overrides[utc_now] = lambda: clock[0]
    h = Harness(app=app, client=TestClient(app), store=store, audit=audit, clock=clock)
    h.as_role(LitellmUserRoles.PROXY_ADMIN)
    h.with_conditions(
        WarningCondition(id="env_credential_login", severity="critical"),
        WarningCondition(id="detailed_debug", severity="warning"),
    )
    yield h


class TestLicenseExpiryCondition:
    @pytest.mark.parametrize(
        ("days_left", "severity"),
        [(31, None), (30, "warning"), (8, "warning"), (7, "critical"), (0, "critical"), (-1, "critical")],
    )
    def test_severity_follows_days_left(self, days_left, severity):
        expires_on = (T0 + timedelta(days=days_left)).date().isoformat()
        condition = license_expiry_condition(expires_on, T0)
        if severity is None:
            assert condition is None
            return
        assert condition == WarningCondition(id=f"license_expiry:{expires_on}", severity=severity)

    def test_days_left_counts_utc_calendar_days(self):
        just_before_midnight = datetime(2026, 10, 14, 23, 59, tzinfo=timezone.utc)
        assert license_expiry_condition("2026-10-21", just_before_midnight).severity == "critical"
        assert license_expiry_condition("2026-10-22", just_before_midnight).severity == "warning"

    def test_renewed_license_gets_a_new_id(self):
        first = license_expiry_condition("2026-10-20", T0)
        renewed = license_expiry_condition("2026-11-10", T0)
        assert first is not None and renewed is not None
        assert first.id != renewed.id

    @pytest.mark.parametrize("expiration_date", [None, "", "not-a-date"])
    def test_missing_or_invalid_date_has_no_warning(self, expiration_date):
        assert license_expiry_condition(expiration_date, T0) is None


class TestDeploymentWarningConditions:
    @staticmethod
    def _conditions(**overrides: object) -> dict[str, WarningCondition]:
        signals: dict[str, object] = {
            "env_credential_login_enabled": False,
            "license_expiration_date": None,
            "no_redis_on_multiple_workers": False,
            "no_redis_warning_disabled_by_env": False,
            "detailed_debug": False,
            "now": T0,
            **overrides,
        }
        return {c.id: c for c in deployment_warning_conditions(**signals)}

    def test_healthy_deployment_has_no_warnings(self):
        assert self._conditions() == {}

    def test_every_signal_maps_to_its_warning_and_severity(self):
        conditions = self._conditions(
            env_credential_login_enabled=True,
            license_expiration_date="2026-11-01",
            no_redis_on_multiple_workers=True,
            detailed_debug=True,
        )
        assert {cid: c.severity for cid, c in conditions.items()} == {
            "env_credential_login": "critical",
            "license_expiry:2026-11-01": "warning",
            "no_redis": "warning",
            "detailed_debug": "warning",
        }
        assert all(c.env_var_ignore is None for c in conditions.values())

    def test_no_redis_env_var_keeps_the_warning_listed_as_env_ignored(self):
        conditions = self._conditions(no_redis_on_multiple_workers=True, no_redis_warning_disabled_by_env=True)
        assert conditions["no_redis"].env_var_ignore == "LITELLM_DISABLE_NO_REDIS_WARNING"

    def test_env_var_alone_does_not_raise_the_no_redis_warning(self):
        assert self._conditions(no_redis_warning_disabled_by_env=True) == {}


@pytest.fixture
def set_logger_level() -> Iterator[Callable[[int], None]]:
    original = verbose_logger.level
    yield verbose_logger.setLevel
    verbose_logger.setLevel(original)


class TestCurrentWarningConditions:
    @staticmethod
    def _proxy_state(monkeypatch, *, redis_cache: object, general_settings: dict[str, object], license: object) -> None:
        monkeypatch.setattr("litellm.proxy.proxy_server.redis_usage_cache", redis_cache)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", general_settings)
        monkeypatch.setattr("litellm.proxy.proxy_server.premium_user_data", license)
        monkeypatch.setattr(
            "litellm.proxy.proxy_server._license_check",
            SimpleNamespace(airgapped_license_data=None, license_str=None, public_key=None),
        )

    @pytest.mark.asyncio
    async def test_reads_every_signal_from_the_running_proxy(self, monkeypatch, set_logger_level):
        self._proxy_state(monkeypatch, redis_cache=None, general_settings={}, license={"expiration_date": "2026-10-20"})
        monkeypatch.setenv("LITELLM_DISABLE_NO_REDIS_WARNING", "true")
        set_logger_level(logging.DEBUG)

        conditions = await current_warning_conditions(T0)

        assert conditions == (
            WarningCondition(id="env_credential_login", severity="critical"),
            WarningCondition(id="license_expiry:2026-10-20", severity="critical"),
            WarningCondition(id="no_redis", severity="warning", env_var_ignore="LITELLM_DISABLE_NO_REDIS_WARNING"),
            WarningCondition(id="detailed_debug", severity="warning"),
        )

    @pytest.mark.asyncio
    async def test_healthy_proxy_has_no_conditions(self, monkeypatch, set_logger_level):
        self._proxy_state(
            monkeypatch,
            redis_cache=SimpleNamespace(),
            general_settings={"disable_env_credential_login": True},
            license=None,
        )
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        set_logger_level(logging.INFO)

        assert await current_warning_conditions(T0) == ()


class TestDeploymentWarningIgnoreStore:
    def test_no_database_means_no_store(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
        assert deployment_warning_ignore_store() is None

    @pytest.mark.asyncio
    async def test_reads_ignores_from_the_writer_not_a_lagging_replica(self, monkeypatch):
        def ui_settings_table(row: object) -> SimpleNamespace:
            async def find_unique(where: dict[str, str]) -> object:
                return row

            return SimpleNamespace(litellm_uisettings=SimpleNamespace(find_unique=find_unique))

        fresh = {"detailed_debug": {"ignored_by": "a", "ignored_at": "2026-10-14T12:00:00Z"}}
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.prisma_client",
            SimpleNamespace(
                db=ui_settings_table(None), writer_db=ui_settings_table(SimpleNamespace(ui_settings=fresh))
            ),
        )
        store = deployment_warning_ignore_store()
        assert store is not None
        assert await store.get_raw_ignores() == fresh


class TestGetDeploymentWarnings:
    @pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
    def test_admins_and_viewers_read_warnings(self, harness, role):
        harness.as_role(role)
        response = harness.client.get("/health/deployment_warnings")
        assert response.status_code == 200
        assert response.json() == {
            "warnings": [
                {"id": "env_credential_login", "severity": "critical", "ignore": None},
                {"id": "detailed_debug", "severity": "warning", "ignore": None},
            ],
            "ignore_available": True,
        }

    @pytest.mark.parametrize(
        "role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, LitellmUserRoles.TEAM]
    )
    def test_other_roles_are_rejected(self, harness, role):
        harness.as_role(role)
        assert harness.client.get("/health/deployment_warnings").status_code == 403

    def test_env_var_ignore_is_reported(self, harness):
        harness.with_conditions(
            WarningCondition(id="no_redis", severity="warning", env_var_ignore="LITELLM_DISABLE_NO_REDIS_WARNING")
        )
        assert harness.warnings()["no_redis"]["ignore"] == {
            "source": "env_var",
            "env_var": "LITELLM_DISABLE_NO_REDIS_WARNING",
        }

    def test_env_var_ignore_wins_over_a_stored_admin_ignore(self, harness):
        harness.with_conditions(WarningCondition(id="no_redis", severity="warning"))
        harness.client.patch("/health/deployment_warnings/no_redis", json={"ignored": True})
        harness.with_conditions(
            WarningCondition(id="no_redis", severity="warning", env_var_ignore="LITELLM_DISABLE_NO_REDIS_WARNING")
        )
        assert harness.warnings()["no_redis"]["ignore"]["source"] == "env_var"

    def test_stored_ignore_without_timezone_reads_as_unreadable(self, harness):
        harness.store.raw = {
            "detailed_debug": {"ignored_by": "a", "ignored_at": "2026-10-14T12:00:00", "ignored_until": None}
        }
        warnings = harness.warnings()
        assert set(warnings) == {"env_credential_login", "detailed_debug"}
        assert warnings["detailed_debug"]["ignore"] is None

    def test_corrupt_stored_ignores_read_as_none(self, harness):
        harness.store.raw = "{not json"
        warnings = harness.warnings()
        assert set(warnings) == {"env_credential_login", "detailed_debug"}
        assert all(w["ignore"] is None for w in warnings.values())

    def test_without_database_ignoring_is_reported_unavailable(self, harness):
        harness.without_database()
        response = harness.client.get("/health/deployment_warnings")
        assert response.status_code == 200
        assert response.json()["ignore_available"] is False
        assert [w["id"] for w in response.json()["warnings"]] == ["env_credential_login", "detailed_debug"]


class TestUpdateDeploymentWarningIgnore:
    def test_snooze_computes_end_time_and_applies_to_every_admin(self, harness):
        harness.as_role(LitellmUserRoles.PROXY_ADMIN, user_id="alice")
        response = harness.client.patch(
            "/health/deployment_warnings/env_credential_login", json={"ignored": True, "snooze_seconds": 86400}
        )
        assert response.status_code == 200
        expected_ignore = {
            "source": "admin",
            "ignored_by": "alice",
            "ignored_at": "2026-10-14T12:00:00Z",
            "ignored_until": "2026-10-15T12:00:00Z",
        }
        assert response.json() == {"id": "env_credential_login", "ignore": expected_ignore}

        harness.as_role(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, user_id="bob")
        warnings = harness.warnings()
        assert warnings["env_credential_login"]["ignore"] == expected_ignore
        assert warnings["detailed_debug"]["ignore"] is None

    def test_snooze_ends_exactly_at_its_end_time(self, harness):
        harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": True, "snooze_seconds": 60})

        harness.clock[0] = T0 + timedelta(seconds=59)
        assert harness.warnings()["detailed_debug"]["ignore"] is not None

        harness.clock[0] = T0 + timedelta(seconds=60)
        assert harness.warnings()["detailed_debug"]["ignore"] is None

    def test_omitting_snooze_ignores_forever(self, harness):
        response = harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": True})
        assert response.status_code == 200
        assert response.json()["ignore"]["ignored_until"] is None

        harness.clock[0] = T0 + timedelta(days=3650)
        assert harness.warnings()["detailed_debug"]["ignore"]["ignored_at"] == "2026-10-14T12:00:00Z"

    def test_restore_clears_only_that_warning(self, harness):
        harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": True})
        harness.client.patch("/health/deployment_warnings/env_credential_login", json={"ignored": True})

        response = harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": False})
        assert response.status_code == 200
        assert response.json() == {"id": "detailed_debug", "ignore": None}

        warnings = harness.warnings()
        assert warnings["detailed_debug"]["ignore"] is None
        assert warnings["env_credential_login"]["ignore"]["source"] == "admin"

    def test_any_severity_can_be_ignored(self, harness):
        harness.with_conditions(WarningCondition(id="license_expiry:2026-10-16", severity="critical"))
        response = harness.client.patch("/health/deployment_warnings/license_expiry:2026-10-16", json={"ignored": True})
        assert response.status_code == 200
        assert harness.warnings()["license_expiry:2026-10-16"]["ignore"]["ignored_by"] == "admin-user"

    def test_ignoring_an_old_license_date_does_not_hide_the_renewed_one(self, harness):
        harness.client.patch("/health/deployment_warnings/license_expiry:2026-10-16", json={"ignored": True})
        harness.with_conditions(WarningCondition(id="license_expiry:2026-11-01", severity="warning"))
        assert harness.warnings()["license_expiry:2026-11-01"]["ignore"] is None

    def test_change_is_written_to_the_config_audit_log(self, harness):
        harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": True})
        harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": False})

        assert len(harness.audit.calls) == 2
        ignore_call, restore_call = harness.audit.calls
        assert ignore_call["table_name"] == LitellmTableNames.UI_SETTINGS_TABLE_NAME
        assert ignore_call["before_value"] == []
        assert ignore_call["after_value"] == [
            {
                "id": "detailed_debug",
                "ignore": {
                    "source": "admin",
                    "ignored_by": "admin-user",
                    "ignored_at": "2026-10-14T12:00:00Z",
                    "ignored_until": None,
                },
            }
        ]
        assert restore_call["before_value"] == ignore_call["after_value"]
        assert restore_call["after_value"] == []
        assert {call["param_name"] for call in harness.audit.calls} == {DEPLOYMENT_WARNING_IGNORES_ROW_ID}
        assert {call["action"] for call in harness.audit.calls} == {"updated"}

    def test_key_without_a_user_is_recorded_as_the_proxy_admin(self, harness):
        harness.as_role(LitellmUserRoles.PROXY_ADMIN, user_id=None)
        response = harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": True})
        assert response.json()["ignore"]["ignored_by"] == LITELLM_PROXY_ADMIN_NAME

    def test_expired_snoozes_are_not_carried_into_the_next_write(self, harness):
        harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": True, "snooze_seconds": 60})
        harness.clock[0] = T0 + timedelta(minutes=5)
        harness.client.patch("/health/deployment_warnings/env_credential_login", json={"ignored": True})

        assert [entry["id"] for entry in harness.audit.calls[-1]["after_value"]] == ["env_credential_login"]

    def test_audit_entry_survives_secret_redaction(self, harness):
        harness.client.patch(
            "/health/deployment_warnings/env_credential_login", json={"ignored": True, "snooze_seconds": 3600}
        )

        stored = json.loads(_dump_redacted_config(harness.audit.calls[-1]["after_value"]))

        assert stored == [
            {
                "id": "env_credential_login",
                "ignore": {
                    "source": "admin",
                    "ignored_by": "admin-user",
                    "ignored_at": "2026-10-14T12:00:00Z",
                    "ignored_until": "2026-10-14T13:00:00Z",
                },
            }
        ]

    @pytest.mark.parametrize("snooze_seconds", [0, -1, THIRTY_DAYS + 1])
    def test_snooze_outside_thirty_days_is_rejected(self, harness, snooze_seconds):
        response = harness.client.patch(
            "/health/deployment_warnings/detailed_debug", json={"ignored": True, "snooze_seconds": snooze_seconds}
        )
        assert response.status_code == 422
        assert harness.store.raw is None

    def test_thirty_day_snooze_is_allowed(self, harness):
        response = harness.client.patch(
            "/health/deployment_warnings/detailed_debug", json={"ignored": True, "snooze_seconds": THIRTY_DAYS}
        )
        assert response.status_code == 200
        assert response.json()["ignore"]["ignored_until"] == "2026-11-13T12:00:00Z"

    def test_snooze_without_ignoring_is_rejected(self, harness):
        response = harness.client.patch(
            "/health/deployment_warnings/detailed_debug", json={"ignored": False, "snooze_seconds": 60}
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("warning_id", ["made_up", "license_expiry:soon", "license_expiry:20261016"])
    def test_unknown_warning_is_rejected(self, harness, warning_id):
        response = harness.client.patch(f"/health/deployment_warnings/{warning_id}", json={"ignored": True})
        assert response.status_code == 404
        assert harness.store.raw is None

    @pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER])
    def test_only_proxy_admins_can_ignore(self, harness, role):
        harness.as_role(role)
        response = harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": True})
        assert response.status_code == 403
        assert harness.store.raw is None
        assert harness.audit.calls == []

    def test_without_database_ignoring_is_unavailable(self, harness):
        harness.without_database()
        response = harness.client.patch("/health/deployment_warnings/detailed_debug", json={"ignored": True})
        assert response.status_code == 503
        assert "database" in response.json()["detail"].lower()
        assert harness.audit.calls == []
