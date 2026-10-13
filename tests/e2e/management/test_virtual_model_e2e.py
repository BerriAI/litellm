"""Live e2e: the virtual model management routes' lifecycle contract.

Each test creates its resources under unique names (deleted on teardown) and
asserts both halves of the contract: the recorded state (the info route reflects
the write) and the enforced behavior (the data plane routes traffic correctly
through the virtual model abstraction).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final, TypeVar

import pytest
from e2e_config import CHEAP_OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, unique_marker
from e2e_http import NoBody, UnknownApiError, unwrap
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import (
    ChatBody,
    ChatMessage,
    EmbedBody,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ResponsesBody,
    ResponsesOutput,
    TeamMemberEntry,
    TeamNewBody,
    UserNewBody,
    VirtualModelCreateBody,
    VirtualModelListResponse,
    VirtualModelMemberRequest,
    VirtualModelPatchBody,
    VirtualModelRow,
)
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.e2e

BACKEND_MODEL: Final = CHEAP_OPENAI_MODEL
_UNREACHABLE_API_BASE: Final = "http://127.0.0.1:9/unreachable"

T = TypeVar("T")


def _poll(proxy: ProxyClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(proxy.poll_interval)
    pytest.fail(failure)


def _create_virtual_model(client: ManagementClient, body: VirtualModelCreateBody) -> VirtualModelRow:
    return unwrap(
        client.proxy.transport.post(
            "/virtual_model",
            headers=client.proxy.transport.master,
            json=body,
            response_type=VirtualModelRow,
        )
    )


def _get_virtual_model(client: ManagementClient, virtual_model_id: str) -> VirtualModelRow:
    return unwrap(
        client.proxy.transport.get(
            f"/virtual_model/{virtual_model_id}",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=VirtualModelRow,
        )
    )


def _delete_virtual_model(client: ManagementClient, virtual_model_id: str) -> VirtualModelRow:
    return unwrap(
        client.proxy.transport.delete(
            f"/virtual_model/{virtual_model_id}",
            headers=client.proxy.transport.master,
            params=NoBody(),
            json=NoBody(),
            response_type=VirtualModelRow,
        )
    )


def _patch_virtual_model(
    client: ManagementClient, virtual_model_id: str, body: VirtualModelPatchBody
) -> VirtualModelRow:
    return unwrap(
        client.proxy.transport.patch(
            f"/virtual_model/{virtual_model_id}",
            headers=client.proxy.transport.master,
            json=body,
            response_type=VirtualModelRow,
        )
    )


class VirtualModelListParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    search: str | None = None


class TeamWithMembersNewBody(TeamNewBody):
    model_config = ConfigDict(frozen=True)

    members_with_roles: tuple[TeamMemberEntry, ...] = ()


class ConfigUpdateBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    router_settings: dict[str, object] | None = None


class ConfigUpdateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str


def _list_virtual_models(client: ManagementClient, search: str | None = None) -> tuple[VirtualModelRow, ...]:
    return unwrap(
        client.proxy.transport.get(
            "/virtual_model",
            headers=client.proxy.transport.master,
            params=VirtualModelListParams(search=search),
            response_type=VirtualModelListResponse,
        )
    ).root


def _virtual_model_entry(
    client: ManagementClient, virtual_model_name: str, search: str | None = None
) -> VirtualModelRow | None:
    return next(
        (vm for vm in _list_virtual_models(client, search=search) if vm.virtual_model_name == virtual_model_name),
        None,
    )


class TestVirtualModelRoutes:
    @pytest.mark.covers("mgmt.virtual_model.create.persists")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_create_persists_with_two_members(self, client: ManagementClient, resources: ResourceManager) -> None:
        vm_name = f"e2e-vm-{unique_marker()}"
        member1_name = f"e2e-vm-member1-{unique_marker()}"
        member2_name = f"e2e-vm-member2-{unique_marker()}"

        member1_id = client.proxy.create_model(
            member1_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member1_id))

        member2_id = client.proxy.create_model(
            member2_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member2_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(
                    VirtualModelMemberRequest(model_id=member1_id, priority=1),
                    VirtualModelMemberRequest(model_id=member2_id, priority=2),
                ),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        assert created.virtual_model_name == vm_name
        assert len(created.members) == 2
        assert created.members[0].model_id == member1_id
        assert created.members[0].priority == 1
        assert created.members[1].model_id == member2_id
        assert created.members[1].priority == 2

        retrieved = _get_virtual_model(client, created.virtual_model_id)
        assert retrieved.virtual_model_name == vm_name
        assert len(retrieved.members) == 2
        assert {m.model_id for m in retrieved.members} == {member1_id, member2_id}

        listed = _virtual_model_entry(client, vm_name)
        assert listed is not None, f"{vm_name} absent from /virtual_model list after creation"
        assert listed.virtual_model_id == created.virtual_model_id

    @pytest.mark.covers("mgmt.virtual_model.create.member_forbidden")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_create_with_nonexistent_model_id_rejected(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        vm_name = f"e2e-vm-nonexistent-{unique_marker()}"
        nonexistent_id = "nonexistent-model-id-12345"

        result = client.proxy.transport.post(
            "/virtual_model",
            headers=client.proxy.transport.master,
            json=VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(VirtualModelMemberRequest(model_id=nonexistent_id, priority=1),),
            ),
            response_type=VirtualModelRow,
        )

        assert isinstance(result, UnknownApiError), (
            f"expected validation error for nonexistent model_id, got {type(result)}"
        )
        assert result.status_code == 400, (
            f"expected 400 for nonexistent model_id, got {result.status_code}: {result.body[:300]}"
        )
        assert nonexistent_id in result.body, (
            f"error body must mention the missing model_id {nonexistent_id}, got: {result.body[:300]}"
        )

        assert _virtual_model_entry(client, vm_name) is None, (
            f"virtual model {vm_name} must not persist after failed creation"
        )

    @pytest.mark.covers(
        "mgmt.virtual_model.update.owner_allowed",
        "mgmt.virtual_model.delete.owner_allowed",
        "mgmt.virtual_model.update.stranger_forbidden",
        "mgmt.virtual_model.delete.stranger_forbidden",
    )
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_owner_manages_own_virtual_model(self, client: ManagementClient, resources: ResourceManager) -> None:
        owner_email = f"e2e-vm-owner-{unique_marker()}@example.com"
        owner_user_id = client.create_user(UserNewBody(user_email=owner_email, user_role="internal_user"))
        resources.defer(lambda: client.delete_user(owner_user_id))

        owner_key = client.proxy.generate_key(
            KeyGenerateBody(models=[], user_id=owner_user_id, key_alias=f"e2e-vm-owner-key-{unique_marker()}")
        )
        resources.defer(lambda: client.proxy.delete_key(owner_key))

        stranger_email = f"e2e-vm-stranger-{unique_marker()}@example.com"
        stranger_user_id = client.create_user(UserNewBody(user_email=stranger_email, user_role="internal_user"))
        resources.defer(lambda: client.delete_user(stranger_user_id))

        stranger_key = client.proxy.generate_key(
            KeyGenerateBody(models=[], user_id=stranger_user_id, key_alias=f"e2e-vm-stranger-key-{unique_marker()}")
        )
        resources.defer(lambda: client.proxy.delete_key(stranger_key))

        vm_name = f"e2e-vm-owner-test-{unique_marker()}"
        member_name = f"e2e-vm-owner-member-{unique_marker()}"
        member_id = client.proxy.create_model(
            member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_id))

        created = unwrap(
            client.proxy.transport.post(
                "/virtual_model",
                headers=client.proxy.transport.bearer(owner_key),
                json=VirtualModelCreateBody(
                    virtual_model_name=vm_name,
                    members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
                ),
                response_type=VirtualModelRow,
            )
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        assert created.created_by == owner_user_id, (
            f"expected created_by to echo owner user_id, got {created.created_by}"
        )

        patched = unwrap(
            client.proxy.transport.patch(
                f"/virtual_model/{created.virtual_model_id}",
                headers=client.proxy.transport.bearer(owner_key),
                json=VirtualModelPatchBody(description="owner update"),
                response_type=VirtualModelRow,
            )
        )
        assert patched.description == "owner update"

        stranger_patch_result = client.proxy.transport.patch(
            f"/virtual_model/{created.virtual_model_id}",
            headers=client.proxy.transport.bearer(stranger_key),
            json=VirtualModelPatchBody(description="stranger update"),
            response_type=VirtualModelRow,
        )
        assert isinstance(stranger_patch_result, UnknownApiError) and stranger_patch_result.status_code == 403, (
            f"expected 403 for stranger PATCH, got {type(stranger_patch_result)}: {stranger_patch_result}"
        )
        assert "owner, a team admin, or a proxy admin" in stranger_patch_result.body, (
            f"expected owner/team-admin/proxy-admin rejection message, got: {stranger_patch_result.body[:300]}"
        )

        stranger_delete_result = client.proxy.transport.delete(
            f"/virtual_model/{created.virtual_model_id}",
            headers=client.proxy.transport.bearer(stranger_key),
            params=NoBody(),
            json=NoBody(),
            response_type=VirtualModelRow,
        )
        assert isinstance(stranger_delete_result, UnknownApiError) and stranger_delete_result.status_code == 403, (
            f"expected 403 for stranger DELETE, got {type(stranger_delete_result)}: {stranger_delete_result}"
        )
        assert "owner, a team admin, or a proxy admin" in stranger_delete_result.body, (
            f"expected owner/team-admin/proxy-admin rejection message, got: {stranger_delete_result.body[:300]}"
        )

        deleted = unwrap(
            client.proxy.transport.delete(
                f"/virtual_model/{created.virtual_model_id}",
                headers=client.proxy.transport.bearer(owner_key),
                params=NoBody(),
                json=NoBody(),
                response_type=VirtualModelRow,
            )
        )
        assert deleted.virtual_model_id == created.virtual_model_id

    @pytest.mark.covers("mgmt.virtual_model.update.team_admin_allowed")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_team_admin_manages_team_virtual_model(self, client: ManagementClient, resources: ResourceManager) -> None:
        admin_email = f"e2e-vm-team-admin-{unique_marker()}@example.com"
        admin_user_id = client.create_user(UserNewBody(user_email=admin_email, user_role="internal_user"))
        resources.defer(lambda: client.delete_user(admin_user_id))

        member_email = f"e2e-vm-team-member-{unique_marker()}@example.com"
        member_user_id = client.create_user(UserNewBody(user_email=member_email, user_role="internal_user"))
        resources.defer(lambda: client.delete_user(member_user_id))

        team_alias = f"e2e-vm-team-{unique_marker()}"
        team_id = client.create_team(
            TeamWithMembersNewBody(
                team_alias=team_alias,
                models=[],
                members_with_roles=(
                    TeamMemberEntry(role="admin", user_id=admin_user_id),
                    TeamMemberEntry(role="user", user_id=member_user_id),
                ),
            )
        )
        resources.defer(lambda: client.delete_team(team_id))

        admin_key = client.proxy.generate_key(
            KeyGenerateBody(models=[], user_id=admin_user_id, team_id=team_id)
        )
        resources.defer(lambda: client.proxy.delete_key(admin_key))

        member_key = client.proxy.generate_key(
            KeyGenerateBody(models=[], user_id=member_user_id, team_id=team_id)
        )
        resources.defer(lambda: client.proxy.delete_key(member_key))

        vm_name = f"e2e-vm-team-test-{unique_marker()}"
        member_model_name = f"e2e-vm-team-member-model-{unique_marker()}"
        member_model_id = client.proxy.create_model(
            member_model_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_model_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(VirtualModelMemberRequest(model_id=member_model_id, priority=1),),
                team_id=team_id,
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        assert created.team_id == team_id

        patched = unwrap(
            client.proxy.transport.patch(
                f"/virtual_model/{created.virtual_model_id}",
                headers=client.proxy.transport.bearer(admin_key),
                json=VirtualModelPatchBody(description="team admin update"),
                response_type=VirtualModelRow,
            )
        )
        assert patched.description == "team admin update"

        member_patch_result = client.proxy.transport.patch(
            f"/virtual_model/{created.virtual_model_id}",
            headers=client.proxy.transport.bearer(member_key),
            json=VirtualModelPatchBody(description="plain member update"),
            response_type=VirtualModelRow,
        )
        assert isinstance(member_patch_result, UnknownApiError) and member_patch_result.status_code == 403, (
            f"expected 403 for plain team member PATCH, got {type(member_patch_result)}: {member_patch_result}"
        )
        assert "owner, a team admin, or a proxy admin" in member_patch_result.body, (
            f"expected owner/team-admin/proxy-admin rejection message, got: {member_patch_result.body[:300]}"
        )

    @pytest.mark.covers("mgmt.virtual_model.list.owner_scoped")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_list_is_owner_scoped(self, client: ManagementClient, resources: ResourceManager) -> None:
        user_email = f"e2e-vm-list-user-{unique_marker()}@example.com"
        user_id = client.create_user(UserNewBody(user_email=user_email, user_role="internal_user"))
        resources.defer(lambda: client.delete_user(user_id))

        user_key = client.proxy.generate_key(
            KeyGenerateBody(models=[], user_id=user_id, key_alias=f"e2e-vm-list-key-{unique_marker()}")
        )
        resources.defer(lambda: client.proxy.delete_key(user_key))

        vm_a_name = f"e2e-vm-list-a-{unique_marker()}"
        vm_b_name = f"e2e-vm-list-b-{unique_marker()}"
        member_name = f"e2e-vm-list-member-{unique_marker()}"

        member_id = client.proxy.create_model(
            member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_id))

        vm_a = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_a_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, vm_a.virtual_model_id))

        vm_b = unwrap(
            client.proxy.transport.post(
                "/virtual_model",
                headers=client.proxy.transport.bearer(user_key),
                json=VirtualModelCreateBody(
                    virtual_model_name=vm_b_name,
                    members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
                ),
                response_type=VirtualModelRow,
            )
        )
        resources.defer(lambda: _delete_virtual_model(client, vm_b.virtual_model_id))

        user_list = unwrap(
            client.proxy.transport.get(
                "/virtual_model",
                headers=client.proxy.transport.bearer(user_key),
                params=VirtualModelListParams(search=None),
                response_type=VirtualModelListResponse,
            )
        ).root

        user_vm_names = {vm.virtual_model_name for vm in user_list}
        assert vm_b_name in user_vm_names, f"user's own VM {vm_b_name} must appear in user's list"
        assert vm_a_name not in user_vm_names, f"admin-created VM {vm_a_name} must not appear in user's list"

        master_list = _list_virtual_models(client)
        master_vm_names = {vm.virtual_model_name for vm in master_list}
        assert vm_a_name in master_vm_names, f"VM {vm_a_name} must appear in master list"
        assert vm_b_name in master_vm_names, f"VM {vm_b_name} must appear in master list"

    @pytest.mark.covers("mgmt.virtual_model.call.happy_path")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
            providers=(Provider.OPENAI,),
            models=(BACKEND_MODEL,),
        )
    )
    def test_call_succeeds_with_healthy_members(self, client: ManagementClient, resources: ResourceManager) -> None:
        vm_name = f"e2e-vm-call-happy-{unique_marker()}"
        member1_name = f"e2e-vm-call-member1-{unique_marker()}"
        member2_name = f"e2e-vm-call-member2-{unique_marker()}"

        member1_id = client.proxy.create_model(
            member1_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member1_id))

        member2_id = client.proxy.create_model(
            member2_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member2_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(
                    VirtualModelMemberRequest(model_id=member1_id, priority=1),
                    VirtualModelMemberRequest(model_id=member2_id, priority=2),
                ),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        response = unwrap(
            client.proxy.chat(
                client.master_key,
                ChatBody(
                    model=f"virtual_model/{vm_name}",
                    messages=[ChatMessage(role="user", content="say ok")],
                    max_tokens=16,
                ),
            )
        )

        assert len(response.choices) > 0, f"response must have choices, got {response}"

    @pytest.mark.covers("mgmt.virtual_model.call.failover")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
            providers=(Provider.OPENAI,),
            models=(BACKEND_MODEL,),
        )
    )
    def test_call_failovers_from_invalid_to_healthy_member(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        vm_name = f"e2e-vm-failover-{unique_marker()}"
        broken_member_name = f"e2e-vm-broken-{unique_marker()}"
        healthy_member_name = f"e2e-vm-healthy-{unique_marker()}"

        broken_member_id = client.proxy.create_model(
            broken_member_name,
            LiteLLMParamsBody(
                model=BACKEND_MODEL,
                api_base=_UNREACHABLE_API_BASE,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(broken_member_id))

        healthy_member_id = client.proxy.create_model(
            healthy_member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(healthy_member_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(
                    VirtualModelMemberRequest(model_id=broken_member_id, priority=1),
                    VirtualModelMemberRequest(model_id=healthy_member_id, priority=2),
                ),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        response = unwrap(
            client.proxy.chat(
                client.master_key,
                ChatBody(
                    model=f"virtual_model/{vm_name}",
                    messages=[ChatMessage(role="user", content="say ok")],
                    max_tokens=16,
                ),
            )
        )

        assert len(response.choices) > 0, (
            f"expected the healthy member to serve after the broken member failed, got {response}"
        )

    @pytest.mark.covers("mgmt.virtual_model.model_delete.cascades")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_member_delete_cascades_to_membership(self, client: ManagementClient, resources: ResourceManager) -> None:
        vm_name = f"e2e-vm-cascade-{unique_marker()}"
        member1_name = f"e2e-vm-cascade-m1-{unique_marker()}"
        member2_name = f"e2e-vm-cascade-m2-{unique_marker()}"

        member1_id = client.proxy.create_model(
            member1_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member1_id))

        member2_id = client.proxy.create_model(
            member2_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member2_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(
                    VirtualModelMemberRequest(model_id=member1_id, priority=1),
                    VirtualModelMemberRequest(model_id=member2_id, priority=2),
                ),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        assert len(created.members) == 2

        client.proxy.delete_model(member1_id)

        def updated_membership() -> VirtualModelRow | None:
            vm = _get_virtual_model(client, created.virtual_model_id)
            return vm if len(vm.members) == 1 else None

        updated = _poll(
            client.proxy,
            updated_membership,
            "membership never shrank after member delete",
        )
        assert len(updated.members) == 1
        assert updated.members[0].model_id == member2_id
        assert updated.live_member_count == 1

    @pytest.mark.covers("mgmt.virtual_model.delete.happy_path")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_delete_removes_virtual_model_and_preserves_members(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        vm_name = f"e2e-vm-delete-{unique_marker()}"
        member_name = f"e2e-vm-delete-member-{unique_marker()}"

        member_id = client.proxy.create_model(
            member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
        )

        deleted = _delete_virtual_model(client, created.virtual_model_id)

        assert deleted.virtual_model_id == created.virtual_model_id
        assert len(deleted.members) == 1

        result = client.proxy.transport.get(
            f"/virtual_model/{created.virtual_model_id}",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=VirtualModelRow,
        )
        assert isinstance(result, UnknownApiError), f"expected 404 after delete, got {type(result)}"
        assert result.status_code == 404, f"expected 404 for deleted virtual model, got {result.status_code}"

        assert _virtual_model_entry(client, vm_name) is None, (
            f"virtual model {vm_name} must not appear in list after deletion"
        )

        member_info = next(
            (entry for entry in client.proxy.model_info() if entry.model_name == member_name),
            None,
        )
        assert member_info is not None, f"member deployment {member_name} must still exist after virtual model deletion"

    @pytest.mark.covers("mgmt.virtual_model.call.embeddings")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
            route=Route.EMBEDDINGS,
            providers=(Provider.OPENAI,),
            models=(OPENAI_EMBEDDING_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_call_embeddings_through_virtual_model(self, client: ManagementClient, resources: ResourceManager) -> None:
        vm_name = f"e2e-vm-embed-{unique_marker()}"
        member_name = f"e2e-vm-embed-member-{unique_marker()}"

        member_id = client.proxy.create_model(
            member_name,
            LiteLLMParamsBody(model=OPENAI_EMBEDDING_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        result = client.proxy.embed(
            client.master_key,
            EmbedBody(
                model=f"virtual_model/{vm_name}",
                input="test input for embedding",
            ),
        )

        response = unwrap(result)
        assert response.model is not None
        assert len(response.data) > 0
        assert len(response.data[0].embedding) > 0
        assert any(v != 0.0 for v in response.data[0].embedding)

    @pytest.mark.covers("mgmt.virtual_model.call.responses")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(BACKEND_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_call_responses_api_through_virtual_model(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        vm_name = f"e2e-vm-responses-{unique_marker()}"
        member_name = f"e2e-vm-responses-member-{unique_marker()}"

        member_id = client.proxy.create_model(
            member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        response = client.proxy.transport.send(
            "/v1/responses",
            headers=client.proxy.transport.bearer(client.master_key),
            json=ResponsesBody(
                model=f"virtual_model/{vm_name}",
                input="say ok",
                max_output_tokens=16,
            ),
        )

        assert response.ok
        assert response.body

        parsed = ResponsesOutput.model_validate_json(response.body)
        assert len(parsed.output) > 0
        assert any(
            item.content and any(content.text and content.text.strip() for content in item.content)
            for item in parsed.output
            if item.type == "message"
        )

    @pytest.mark.covers("mgmt.virtual_model.call.stream")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.OPENAI,),
            models=(BACKEND_MODEL,),
            mode=Mode.STREAM,
        )
    )
    def test_streaming_chat_through_virtual_model(self, client: ManagementClient, resources: ResourceManager) -> None:
        vm_name = f"e2e-vm-stream-{unique_marker()}"
        member_name = f"e2e-vm-stream-member-{unique_marker()}"

        member_id = client.proxy.create_model(
            member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        chunks = client.proxy.chat_stream(
            client.master_key,
            ChatBody(
                model=f"virtual_model/{vm_name}",
                messages=[ChatMessage(role="user", content="say ok")],
                max_tokens=16,
                stream=True,
            ),
        )

        assert chunks.ok
        assert not chunks.stream_error
        assert chunks.chunks > 0
        assert chunks.stream_done

    @pytest.mark.covers("mgmt.virtual_model.list.search")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_list_with_search_filters_by_name(self, client: ManagementClient, resources: ResourceManager) -> None:
        vm1_name = f"e2e-vm-search-alpha-{unique_marker()}"
        vm2_name = f"e2e-vm-search-beta-{unique_marker()}"
        member_name = f"e2e-vm-search-member-{unique_marker()}"

        member_id = client.proxy.create_model(
            member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_id))

        vm1 = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm1_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, vm1.virtual_model_id))

        vm2 = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm2_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, vm2.virtual_model_id))

        alpha_lower = _list_virtual_models(client, search="alpha")
        assert len(alpha_lower) == 1, f"expected one match for 'alpha', got {len(alpha_lower)}"
        assert alpha_lower[0].virtual_model_name == vm1_name

        alpha_upper = _list_virtual_models(client, search="ALPHA")
        assert len(alpha_upper) == 1, f"expected one match for 'ALPHA', got {len(alpha_upper)}"
        assert alpha_upper[0].virtual_model_name == vm1_name

        alpha_partial = _list_virtual_models(client, search="lph")
        assert len(alpha_partial) == 1, f"expected one match for 'lph', got {len(alpha_partial)}"
        assert alpha_partial[0].virtual_model_name == vm1_name

        nonexistent = _list_virtual_models(client, search="nonexistent-marker-xyz")
        assert len(nonexistent) == 0, f"expected no matches for 'nonexistent-marker-xyz', got {len(nonexistent)}"

    @pytest.mark.covers("mgmt.virtual_model.update.remove_member_reroutes")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
            providers=(Provider.OPENAI,),
            models=(BACKEND_MODEL,),
        )
    )
    def test_remove_member_then_call_succeeds_via_survivor(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        vm_name = f"e2e-vm-remove-reroute-{unique_marker()}"
        member1_name = f"e2e-vm-remove-member1-{unique_marker()}"
        member2_name = f"e2e-vm-remove-member2-{unique_marker()}"

        member1_id = client.proxy.create_model(
            member1_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member1_id))

        member2_id = client.proxy.create_model(
            member2_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member2_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(
                    VirtualModelMemberRequest(model_id=member1_id, priority=1),
                    VirtualModelMemberRequest(model_id=member2_id, priority=2),
                ),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        assert len(created.members) == 2

        updated = _patch_virtual_model(
            client,
            created.virtual_model_id,
            VirtualModelPatchBody(remove_members=(member1_id,)),
        )
        assert len(updated.members) == 1, (
            f"expected membership to shrink to one after removal, got {len(updated.members)}"
        )
        assert updated.members[0].model_id == member2_id

        response = unwrap(
            client.proxy.chat(
                client.master_key,
                ChatBody(
                    model=f"virtual_model/{vm_name}",
                    messages=[ChatMessage(role="user", content="say ok")],
                    max_tokens=16,
                ),
            )
        )

        assert len(response.choices) > 0, (
            f"expected the surviving member to serve after priority-1 removal, got {response}"
        )

    @pytest.mark.covers("mgmt.virtual_model.call.retry_ordering")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
            providers=(Provider.OPENAI,),
            models=(BACKEND_MODEL,),
        )
    )
    def test_broken_priority_1_retries_to_priority_2(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        vm_name = f"e2e-vm-retry-order-{unique_marker()}"
        broken_member_name = f"e2e-vm-retry-broken-{unique_marker()}"
        healthy_member_name = f"e2e-vm-retry-healthy-{unique_marker()}"

        broken_member_id = client.proxy.create_model(
            broken_member_name,
            LiteLLMParamsBody(
                model=BACKEND_MODEL,
                api_base=_UNREACHABLE_API_BASE,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(broken_member_id))

        healthy_member_id = client.proxy.create_model(
            healthy_member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(healthy_member_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(
                    VirtualModelMemberRequest(model_id=broken_member_id, priority=1),
                    VirtualModelMemberRequest(model_id=healthy_member_id, priority=2),
                ),
                allowed_fails=100,
                num_retries=1,
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        response = unwrap(
            client.proxy.chat(
                client.master_key,
                ChatBody(
                    model=f"virtual_model/{vm_name}",
                    messages=[ChatMessage(role="user", content="say ok")],
                    max_tokens=16,
                ),
            )
        )

        assert len(response.choices) > 0, (
            f"expected priority-2 healthy member to serve after priority-1 failed, got {response}"
        )

    @pytest.mark.covers("mgmt.virtual_model.call.alias_strategy")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
            providers=(Provider.OPENAI,),
            models=(BACKEND_MODEL,),
        )
    )
    def test_alias_applies_virtual_model_routing_strategy(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        vm_name = f"e2e-vm-alias-strat-{unique_marker()}"
        alias_name = f"e2e-vm-alias-{unique_marker()}"
        member1_name = f"e2e-vm-alias-member1-{unique_marker()}"
        member2_name = f"e2e-vm-alias-member2-{unique_marker()}"

        member1_id = client.proxy.create_model(
            member1_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member1_id))

        member2_id = client.proxy.create_model(
            member2_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member2_id))

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(
                    VirtualModelMemberRequest(model_id=member1_id, priority=1),
                    VirtualModelMemberRequest(model_id=member2_id, priority=2),
                ),
                routing_strategy="simple-shuffle",
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        alias_patch = unwrap(
            client.proxy.transport.post(
                "/config/update",
                headers=client.proxy.transport.master,
                json=ConfigUpdateBody(
                    router_settings={
                        "model_group_alias": {
                            alias_name: f"virtual_model/{vm_name}",
                        },
                    }
                ),
                response_type=ConfigUpdateResponse,
            )
        )
        assert alias_patch.message, f"expected success message, got {alias_patch}"

        response = unwrap(
            client.proxy.chat(
                client.master_key,
                ChatBody(
                    model=alias_name,
                    messages=[ChatMessage(role="user", content="say ok")],
                    max_tokens=16,
                ),
            )
        )

        assert len(response.choices) > 0, f"expected the alias to route through the virtual model, got {response}"
