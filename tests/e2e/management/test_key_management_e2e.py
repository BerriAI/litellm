"""Live e2e: the /key management routes' persistence, health, bulk-update, and
admin-only contracts.

Each test creates its keys under the master key with unique aliases (deleted on
teardown) and asserts the real contract: the info route reflects the write
(persistence), the health route reports the calling key, bulk_update applies to
the target key, and the write routes refuse a non-admin caller. Key writes reach
the auth cache eventually, so the read-backs poll to a deadline instead of
asserting once.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

import pytest

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import KeyDeleteBody, KeyGenerateBody, KeyUpdateBody
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


class KeyToggleBlockBody(BaseModel):
    key: str


class LoggingCallbackStatus(BaseModel):
    callbacks: list[str] | None = None
    status: str | None = None
    details: str | None = None


class KeyHealthResponse(BaseModel):
    key: Literal["healthy", "unhealthy"]
    logging_callbacks: LoggingCallbackStatus | None = None


class BulkKeyUpdateItem(BaseModel):
    key: str
    max_budget: float | None = None


class BulkKeyUpdateBody(BaseModel):
    keys: list[BulkKeyUpdateItem]


class BulkKeyUpdateSuccess(BaseModel):
    key: str


class BulkKeyUpdateFailure(BaseModel):
    key: str
    failed_reason: str


class BulkKeyUpdateResponse(BaseModel):
    total_requested: int
    successful_updates: list[BulkKeyUpdateSuccess]
    failed_updates: list[BulkKeyUpdateFailure]


def _poll[T](client: ManagementClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + client.proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(client.proxy.poll_interval)
    pytest.fail(failure)


def _generate_key(client: ManagementClient, resources: ResourceManager, body: KeyGenerateBody) -> str:
    key = client.proxy.generate_key(body)
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


def _block(client: ManagementClient, key: str) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/key/block",
            headers=client.proxy.transport.master,
            json=KeyToggleBlockBody(key=key),
            response_type=NoBody,
        )
    )


def _unblock(client: ManagementClient, key: str) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/key/unblock",
            headers=client.proxy.transport.master,
            json=KeyToggleBlockBody(key=key),
            response_type=NoBody,
        )
    )


class TestKeyManagementRoutes:
    @pytest.mark.covers("mgmt.key.info.persists")
    def test_info_reflects_the_fields_the_key_was_created_with(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-mgmt-keyinfo-{unique_marker()}"
        key = _generate_key(
            client,
            resources,
            KeyGenerateBody(
                models=["gpt-5.5", "gemini-2.5-flash"],
                key_alias=alias,
                tpm_limit=131313,
                rpm_limit=141414,
            ),
        )

        info = client.proxy.key_info(key)
        assert info.key_alias == alias, f"/key/info reports key_alias {info.key_alias!r}, configured {alias!r}"
        assert info.models == ["gpt-5.5", "gemini-2.5-flash"], (
            f"/key/info reports models {info.models}, configured ['gpt-5.5', 'gemini-2.5-flash']"
        )
        assert info.tpm_limit == 131313, f"/key/info reports tpm_limit {info.tpm_limit}, configured 131313"
        assert info.rpm_limit == 141414, f"/key/info reports rpm_limit {info.rpm_limit}, configured 141414"

    @pytest.mark.covers("mgmt.key.unblock.persists")
    def test_unblock_flips_key_info_blocked_back(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"]))

        _block(client, key)
        _ = _poll(
            client,
            lambda: True if client.proxy.key_info(key).blocked else None,
            "/key/info never reported the key blocked after /key/block before the deadline",
        )

        _unblock(client, key)
        _ = _poll(
            client,
            lambda: True if client.proxy.key_info(key).blocked is False else None,
            "/key/info never reported the key unblocked after /key/unblock before the deadline",
        )

    @pytest.mark.covers("mgmt.key.health.happy_path")
    def test_health_reports_the_calling_key_healthy(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"]))

        health = unwrap(
            client.proxy.transport.post(
                "/key/health",
                headers=client.proxy.transport.bearer(key),
                json=NoBody(),
                response_type=KeyHealthResponse,
            )
        )
        assert health.key == "healthy", f"/key/health reports {health.key!r} for a key with no logging configured"
        assert health.logging_callbacks is None, (
            f"/key/health reports logging_callbacks {health.logging_callbacks!r} for a key with no logging configured"
        )

    @pytest.mark.covers("mgmt.key.bulk_update.happy_path")
    def test_bulk_update_applies_max_budget_to_target_key(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"], max_budget=5.0))
        assert client.proxy.key_info(key).max_budget == 5.0, (
            f"/key/info reports max_budget {client.proxy.key_info(key).max_budget}, configured 5.0"
        )

        result = unwrap(
            client.proxy.transport.post(
                "/key/bulk_update",
                headers=client.proxy.transport.master,
                json=BulkKeyUpdateBody(keys=[BulkKeyUpdateItem(key=key, max_budget=42.0)]),
                response_type=BulkKeyUpdateResponse,
            )
        )
        assert result.total_requested == 1, f"/key/bulk_update reports total_requested {result.total_requested}, sent 1"
        assert result.failed_updates == [], f"/key/bulk_update reported failed updates: {result.failed_updates}"
        assert [entry.key for entry in result.successful_updates] == [key], (
            f"/key/bulk_update successful_updates {[entry.key for entry in result.successful_updates]} did not target {key}"
        )

        _ = _poll(
            client,
            lambda: True if client.proxy.key_info(key).max_budget == 42.0 else None,
            "/key/info never reported max_budget 42.0 after /key/bulk_update before the deadline",
        )

    @pytest.mark.covers("mgmt.key.generate.admin_only")
    def test_generate_forbidden_for_non_admin_key(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        nonadmin = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"]))

        outcome = client.proxy.transport.send(
            "/key/generate",
            headers=client.proxy.transport.bearer(nonadmin),
            json=KeyGenerateBody(models=["gpt-5.5"], key_alias=f"e2e-mgmt-forbidden-{unique_marker()}"),
        )
        assert outcome.status_code in (401, 403), (
            f"non-admin key POSTing /key/generate must be denied 401/403, got {outcome.status_code}: {outcome.body[:300]}"
        )

    @pytest.mark.covers("mgmt.key.delete.admin_only")
    def test_delete_forbidden_for_non_admin_key(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        nonadmin = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"]))
        victim = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"]))

        outcome = client.proxy.transport.send(
            "/key/delete",
            headers=client.proxy.transport.bearer(nonadmin),
            json=KeyDeleteBody(keys=[victim]),
        )
        assert outcome.status_code in (401, 403), (
            f"non-admin key POSTing /key/delete must be denied 401/403, got {outcome.status_code}: {outcome.body[:300]}"
        )
        assert client.proxy.key_info(victim).blocked in (None, False), (
            "victim key should be unaffected by the denied /key/delete"
        )

    @pytest.mark.covers("mgmt.key.update.admin_only")
    def test_update_forbidden_for_non_admin_key(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        nonadmin = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"]))
        target = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"]))

        outcome = client.proxy.transport.send(
            "/key/update",
            headers=client.proxy.transport.bearer(nonadmin),
            json=KeyUpdateBody(key=target, models=["gemini-2.5-flash"]),
        )
        assert outcome.status_code in (401, 403), (
            f"non-admin key POSTing /key/update must be denied 401/403, got {outcome.status_code}: {outcome.body[:300]}"
        )
        assert client.proxy.key_info(target).models == ["gpt-5.5"], (
            f"target key models changed to {client.proxy.key_info(target).models} despite the denied /key/update"
        )
