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
from collections.abc import Callable, Iterator
from typing import Final, Literal

import pytest

from e2e_config import unique_marker
from e2e_http import NoBody, StreamingResponse, unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import (
    CLEAR, ChatResponse, KeyDeleteBody, KeyGenerateBody, KeyInfo, KeyUpdateBody,
    LiteLLMParamsBody, OrgNewBody, TeamNewBody,
)
from pydantic import BaseModel, RootModel

pytestmark = pytest.mark.e2e

TINY_BUDGET = 3e-6
SPEND_MODEL = "claude-haiku-4-5"


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


def _is_budget_block(outcome: StreamingResponse) -> bool:
    return not outcome.ok and "budget_exceeded" in outcome.body


def _spend_until_budget_blocks(client: ManagementClient, key: str) -> None:
    for _ in range(40):
        outcome = client.chat_status(key, SPEND_MODEL, f"spend {unique_marker()}")
        if _is_budget_block(outcome):
            assert outcome.status_code == 429, (
                f"budget refusal must be 429, got {outcome.status_code}: {outcome.body[:200]}"
            )
            return
        assert outcome.ok, f"paid call failed before the budget tripped ({outcome.status_code}): {outcome.body[:300]}"
        time.sleep(2)
    pytest.fail(f"max_budget={TINY_BUDGET} never blocked a call on the key")


def _settled_spend(client: ManagementClient, key: str) -> float | None:
    first = client.proxy.key_info(key).spend or 0.0
    time.sleep(client.proxy.poll_interval)
    second = client.proxy.key_info(key).spend or 0.0
    return second if first > 0 and first == second else None


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


class ProjectIdentity(BaseModel):
    project_id: str


class ProjectCreateBody(BaseModel):
    team_id: str
    project_alias: str
    models: list[str]


class ProjectBlockBody(ProjectIdentity):
    blocked: bool


class ProjectDeleteBody(BaseModel):
    project_ids: list[str]


@pytest.fixture
def project_resources(client: ManagementClient) -> Iterator[ResourceManager]:
    manager: Final = ResourceManager(client=client.proxy, strict_cleanup=True)
    yield manager
    manager.teardown()


class TestKeyManagementRoutes:
    @pytest.mark.covers("mgmt.key.update.persists")
    def test_project_detachment_preserves_key_scope_and_refreshes_auth(
        self, client: ManagementClient, project_resources: ResourceManager
    ) -> None:
        resources: Final = project_resources
        name: Final = f"e2e-detach-{unique_marker()}"
        model_id: Final = client.proxy.create_model(
            name, LiteLLMParamsBody(model="openai/synthetic-detachment", api_key="synthetic", mock_response="orbit")
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        org_id: Final = client.create_org(OrgNewBody(organization_alias=name, models=[name]))
        resources.defer(lambda: client.delete_org(org_id))
        team_id: Final = client.create_team(TeamNewBody(team_alias=name, organization_id=org_id, models=[name]))
        resources.defer(lambda: client.delete_team(team_id))
        project: Final = unwrap(client.proxy.transport.post(
            "/project/new", headers=client.proxy.transport.master,
            json=ProjectCreateBody(team_id=team_id, project_alias=name, models=[name]),
            response_type=ProjectIdentity,
        ))
        resources.defer(lambda: unwrap(client.proxy.transport.delete(
            "/project/delete", headers=client.proxy.transport.master,
            json=ProjectDeleteBody(project_ids=[project.project_id]), response_type=RootModel[list[ProjectIdentity]],
        )))
        key: Final = _generate_key(client, resources, KeyGenerateBody(
            key_alias=name, team_id=team_id, organization_id=org_id, project_id=project.project_id,
            models=[name], max_budget=5, tpm_limit=12345, rpm_limit=97,
        ))
        initial: Final = client.chat_status(key, name, "project attached")
        assert initial.ok, initial.body
        _ = unwrap(client.update_key(KeyUpdateBody(key=key, key_alias=f"{name}-saved")))
        assert client.proxy.key_info(key).project_id == project.project_id
        _ = unwrap(client.update_key(KeyUpdateBody(key=key, project_id=project.project_id)))
        rejected: Final = client.proxy.transport.send(
            "/key/update", headers=client.proxy.transport.master,
            json=KeyUpdateBody(key=key, project_id=f"{name}-different"),
        )
        assert rejected.status_code == 400 and "reassignment" in rejected.body
        assert client.proxy.key_info(key).project_id == project.project_id
        _ = unwrap(client.proxy.transport.post(
            "/project/update", headers=client.proxy.transport.master,
            json=ProjectBlockBody(project_id=project.project_id, blocked=True), response_type=NoBody,
        ))
        blocked: Final = client.chat_status(key, name, "project blocked")
        assert not blocked.ok and "is blocked" in blocked.body
        detached: Final = unwrap(client.proxy.transport.post(
            "/key/update", headers=client.proxy.transport.master,
            json=KeyUpdateBody(key=key, project_id=CLEAR), response_type=KeyInfo,
        ))
        assert detached.project_id is None
        saved: Final = client.proxy.key_info(key)
        assert (saved.project_id, saved.team_id, saved.organization_id) == (None, team_id, org_id)
        assert (saved.models, saved.max_budget, saved.tpm_limit, saved.rpm_limit) == ([name], 5, 12345, 97)
        allowed: Final = client.chat_status(key, name, "project detached")
        assert allowed.ok, allowed.body
        message: Final = ChatResponse.model_validate_json(allowed.body).choices[0].message
        assert message is not None and message.content == "orbit"
        _ = unwrap(client.update_key(KeyUpdateBody(key=key, project_id=CLEAR)))
        assert client.proxy.key_info(key).project_id is None
        denied: Final = client.chat_status(key, f"{name}-outside", "outside key scope")
        assert denied.status_code in (401, 403), denied.body

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

    @pytest.mark.covers("other.key_mgmt.spend_reset.resets_to_value")
    def test_reset_spend_zeroes_recorded_spend_and_lifts_the_budget_block(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = _generate_key(client, resources, KeyGenerateBody(models=[SPEND_MODEL], max_budget=TINY_BUDGET))
        _spend_until_budget_blocks(client, key)
        recorded = _poll(
            client, lambda: _settled_spend(client, key), "key spend never landed in /key/info before the deadline"
        )

        reset = client.reset_key_spend(key, reset_to=0.0)
        assert reset.previous_spend == recorded, (
            f"reset_spend reported previous_spend {reset.previous_spend}, /key/info had recorded {recorded}"
        )
        assert reset.spend == 0.0, f"reset_spend to 0 reported spend {reset.spend}"
        assert client.proxy.key_info(key).spend == 0.0, "/key/info still reports spend after the reset to 0"

        def call_allowed_again() -> bool | None:
            outcome = client.chat_status(key, SPEND_MODEL, f"after reset {unique_marker()}")
            if _is_budget_block(outcome):
                return None
            assert outcome.ok, f"post-reset call failed ({outcome.status_code}): {outcome.body[:300]}"
            return True

        _ = _poll(client, call_allowed_again, "the key stayed budget-blocked after its spend was reset to 0")

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
