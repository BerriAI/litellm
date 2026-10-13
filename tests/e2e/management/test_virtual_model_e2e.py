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
from e2e_http import NoBody, UnauthorizedError, UnknownApiError, unwrap
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
)
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict, RootModel

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


class VirtualModelMemberRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    priority: int = 1


class VirtualModelCreateBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    virtual_model_name: str
    members: tuple[VirtualModelMemberRequest, ...]
    description: str | None = None


class VirtualModelPatchBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    virtual_model_name: str | None = None
    description: str | None = None
    add_members: tuple[VirtualModelMemberRequest, ...] | None = None
    remove_members: tuple[str, ...] | None = None


class VirtualModelMemberRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    priority: int
    model_name: str | None = None


class VirtualModelRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    virtual_model_id: str
    virtual_model_name: str
    description: str | None = None
    members: tuple[VirtualModelMemberRow, ...]
    routing_strategy: str
    live_member_count: int


class VirtualModelListResponse(RootModel[tuple[VirtualModelRow, ...]]):
    pass


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


def _list_virtual_models(client: ManagementClient) -> tuple[VirtualModelRow, ...]:
    return unwrap(
        client.proxy.transport.get(
            "/virtual_model",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=VirtualModelListResponse,
        )
    ).root


def _virtual_model_entry(client: ManagementClient, virtual_model_name: str) -> VirtualModelRow | None:
    return next(
        (vm for vm in _list_virtual_models(client) if vm.virtual_model_name == virtual_model_name),
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

    @pytest.mark.covers("mgmt.virtual_model.update.admin_only")
    @meta(
        Subject(
            domain=Domain.MANAGEMENT,
        )
    )
    def test_non_admin_forbidden_from_mutations(self, client: ManagementClient, resources: ResourceManager) -> None:
        non_admin_key = unwrap(
            client.generate_key(KeyGenerateBody(models=[], key_alias=f"e2e-non-admin-{unique_marker()}"))
        ).key
        resources.defer(lambda: client.proxy.delete_key(non_admin_key))

        vm_name = f"e2e-vm-admin-test-{unique_marker()}"
        member_name = f"e2e-vm-admin-member-{unique_marker()}"
        member_id = client.proxy.create_model(
            member_name,
            LiteLLMParamsBody(model=BACKEND_MODEL),
        )
        resources.defer(lambda: client.proxy.delete_model(member_id))

        result = client.proxy.transport.post(
            "/virtual_model",
            headers=client.proxy.transport.bearer(non_admin_key),
            json=VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
            response_type=VirtualModelRow,
        )

        assert isinstance(result, UnauthorizedError), f"expected auth error for non-admin, got {type(result)}"
        assert "proxy admin" in result.body, (
            f"expected the admin-only rejection for non-admin POST /virtual_model, got: {result.body[:300]}"
        )

        created = _create_virtual_model(
            client,
            VirtualModelCreateBody(
                virtual_model_name=vm_name,
                members=(VirtualModelMemberRequest(model_id=member_id, priority=1),),
            ),
        )
        resources.defer(lambda: _delete_virtual_model(client, created.virtual_model_id))

        patch_result = client.proxy.transport.patch(
            f"/virtual_model/{created.virtual_model_id}",
            headers=client.proxy.transport.bearer(non_admin_key),
            json=VirtualModelPatchBody(description="non-admin update"),
            response_type=VirtualModelRow,
        )
        assert isinstance(patch_result, UnauthorizedError), (
            f"expected auth error for non-admin PATCH, got {type(patch_result)}"
        )
        assert "proxy admin" in patch_result.body, (
            f"expected the admin-only rejection for non-admin PATCH, got: {patch_result.body[:300]}"
        )

        delete_result = client.proxy.transport.delete(
            f"/virtual_model/{created.virtual_model_id}",
            headers=client.proxy.transport.bearer(non_admin_key),
            params=NoBody(),
            json=NoBody(),
            response_type=VirtualModelRow,
        )
        assert isinstance(delete_result, UnauthorizedError), (
            f"expected auth error for non-admin DELETE, got {type(delete_result)}"
        )
        assert "proxy admin" in delete_result.body, (
            f"expected the admin-only rejection for non-admin DELETE, got: {delete_result.body[:300]}"
        )

        body = VirtualModelPatchBody(remove_members=(member_id,))
        admin_removal_result = client.proxy.transport.patch(
            f"/virtual_model/{created.virtual_model_id}",
            headers=client.proxy.transport.master,
            json=body,
            response_type=VirtualModelRow,
        )
        assert isinstance(admin_removal_result, VirtualModelRow), (
            f"expected success for admin removal payload, got {type(admin_removal_result)}"
        )
        assert len(admin_removal_result.members) == 0, (
            f"expected membership to shrink to zero, got {len(admin_removal_result.members)} members"
        )

        non_admin_removal = client.proxy.transport.patch(
            f"/virtual_model/{created.virtual_model_id}",
            headers=client.proxy.transport.bearer(non_admin_key),
            json=body,
            response_type=VirtualModelRow,
        )
        assert isinstance(non_admin_removal, UnauthorizedError), (
            f"expected auth error for non-admin removal, got {type(non_admin_removal)}"
        )

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
