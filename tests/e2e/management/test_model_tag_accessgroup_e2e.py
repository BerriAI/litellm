"""Live e2e: the model, tag, and model-access-group management routes.

Each test creates its resources under unique names (deleted on teardown) and
asserts the route's contract against a live proxy: the admin-only guard on
adding a global model, the tag inventory round-trip through /tag/list and
/tag/delete, and creating a model access group then reading it back through
/access_group/{name}/info. Reads that lag a write poll to a deadline instead of
asserting once.

Request bodies for /model/new are the shared pydantic models; every response
this suite reads is modelled locally so the file is self-contained and no
untyped dict crosses the boundary.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from pydantic import BaseModel, ConfigDict, RootModel

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import KeyGenerateBody, LiteLLMParamsBody, ModelInfoBody, ModelNewBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

_MODEL_PERMISSION_DENIED_MARKER = "does not have permission to make this model call"
_DUMMY_MODEL = "openai/gpt-5.5"
_DUMMY_API_KEY = "e2e-dummy-key"


def _poll[T](proxy: ProxyClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(proxy.poll_interval)
    pytest.fail(failure)


# ---------- tag route models / helpers ----------


class TagCreateBody(BaseModel):
    name: str
    description: str | None = None


class TagDeleteBody(BaseModel):
    name: str


class TagEntry(BaseModel):
    name: str
    description: str | None = None


class TagCatalog(RootModel[list[TagEntry]]):
    """GET /tag/list answers with a bare array of tag configs, not an object
    wrapping them; read the rows off .root."""


def _tag_list(client: ManagementClient) -> tuple[TagEntry, ...]:
    return tuple(
        unwrap(
            client.proxy.transport.get(
                "/tag/list",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=TagCatalog,
            )
        ).root
    )


def _create_tag(client: ManagementClient, body: TagCreateBody) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/tag/new",
            headers=client.proxy.transport.master,
            json=body,
            response_type=NoBody,
        )
    )


def _delete_tag(client: ManagementClient, name: str) -> None:
    """Best-effort delete for teardown: a repeat /tag/delete on an already-deleted
    tag is a no-op the warn-only teardown absorbs."""
    _ = client.proxy.transport.post(
        "/tag/delete",
        headers=client.proxy.transport.master,
        json=TagDeleteBody(name=name),
        response_type=NoBody,
    )


def _delete_tag_strict(client: ManagementClient, name: str) -> None:
    """Strict delete for the act phase: a failed /tag/delete is a hard failure."""
    _ = unwrap(
        client.proxy.transport.post(
            "/tag/delete",
            headers=client.proxy.transport.master,
            json=TagDeleteBody(name=name),
            response_type=NoBody,
        )
    )


# ---------- access group route models / helpers ----------


class AccessGroupNewBody(BaseModel):
    access_group: str
    model_names: list[str]


class AccessGroupNewResponse(BaseModel):
    access_group: str
    models_updated: int


class AccessGroupInfoResponse(BaseModel):
    access_group: str
    model_names: list[str]
    deployment_count: int


def _create_access_group(client: ManagementClient, body: AccessGroupNewBody) -> AccessGroupNewResponse:
    return unwrap(
        client.proxy.transport.post(
            "/access_group/new",
            headers=client.proxy.transport.master,
            json=body,
            response_type=AccessGroupNewResponse,
        )
    )


def _access_group_info(client: ManagementClient, access_group: str) -> AccessGroupInfoResponse | None:
    result = client.proxy.transport.get(
        f"/access_group/{access_group}/info",
        headers=client.proxy.transport.master,
        params=NoBody(),
        response_type=AccessGroupInfoResponse,
    )
    return unwrap(result) if result.kind == "success" else None


def _delete_access_group(client: ManagementClient, access_group: str) -> None:
    """Best-effort delete for teardown; deleting the model behind it removes the
    access group too, so a repeat delete is a no-op the teardown absorbs."""
    _ = client.proxy.transport.delete(
        f"/access_group/{access_group}/delete",
        headers=client.proxy.transport.master,
        json=NoBody(),
        response_type=NoBody,
    )


def _create_db_model(client: ManagementClient, resources: ResourceManager, model_name: str) -> str:
    model_id = client.proxy.create_model(
        model_name, LiteLLMParamsBody(model=_DUMMY_MODEL, api_key=_DUMMY_API_KEY)
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model_id


# ---------- model block route models / helpers ----------


class ModelBlockBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str


class ModelInfoBlockDetail(BaseModel):
    id: str | None = None
    blocked: bool | None = None


class ModelInfoBlockEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_info: ModelInfoBlockDetail = ModelInfoBlockDetail()


class ModelInfoCatalog(BaseModel):
    data: list[ModelInfoBlockEntry] = []


def _model_blocked_flag(client: ManagementClient, model_id: str) -> bool | None:
    catalog = unwrap(
        client.proxy.transport.get(
            "/model/info",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=ModelInfoCatalog,
        )
    )
    entry = next((row for row in catalog.data if row.model_info.id == model_id), None)
    return entry.model_info.blocked if entry is not None else None


class TestModelRoutes:
    @pytest.mark.covers("mgmt.model.add.admin_only")
    def test_non_admin_key_cannot_add_global_model(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = client.proxy.generate_key(KeyGenerateBody(models=[]))
        resources.defer(lambda: client.proxy.delete_key(key))

        model_name = f"e2e-mgmt-model-forbidden-{unique_marker()}"
        outcome = client.proxy.transport.send(
            "/model/new",
            headers=client.proxy.transport.bearer(key),
            json=ModelNewBody(
                model_name=model_name,
                litellm_params=LiteLLMParamsBody(model=_DUMMY_MODEL, api_key=_DUMMY_API_KEY),
                model_info=ModelInfoBody(),
            ),
        )

        assert outcome.status_code == 403, (
            f"non-admin key adding a global model (no team_id) must be denied 403, got "
            f"{outcome.status_code}: {outcome.body[:300]}"
        )
        assert _MODEL_PERMISSION_DENIED_MARKER in outcome.body, (
            f"403 body must be the model-permission denial, got: {outcome.body[:300]}"
        )

        cataloged = [entry.model_name for entry in client.proxy.model_info()]
        assert model_name not in cataloged, (
            f"{model_name!r} was registered in /model/info despite the 403; the admin-only "
            f"guard did not block the write"
        )

    @pytest.mark.covers("mgmt.model.block.persists")
    def test_block_then_unblock_persists_to_model_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        """The blocked flag's persistence is read back from /model/info, not from the
        /model/block response: that route currently returns a non-2xx serialization
        envelope even though the DB write lands, so the /model/info read-back is the
        authoritative persistence contract and keeps this test valid once the
        response shape is fixed."""
        model_name = f"e2e-mgmt-model-block-{unique_marker()}"
        model_id = _create_db_model(client, resources, model_name)

        assert _model_blocked_flag(client, model_id) is not True, (
            f"{model_name!r} already reports blocked in /model/info before /model/block ran"
        )

        _ = client.proxy.transport.send(
            "/model/block",
            headers=client.proxy.transport.master,
            json=ModelBlockBody(model_id=model_id),
        )
        _ = _poll(
            client.proxy,
            lambda: True if _model_blocked_flag(client, model_id) is True else None,
            f"/model/info never reported {model_name!r} blocked after /model/block",
        )

        _ = client.proxy.transport.send(
            "/model/unblock",
            headers=client.proxy.transport.master,
            json=ModelBlockBody(model_id=model_id),
        )
        _ = _poll(
            client.proxy,
            lambda: True if _model_blocked_flag(client, model_id) is not True else None,
            f"/model/info never cleared blocked for {model_name!r} after /model/unblock",
        )


class TestTagRoutes:
    @pytest.mark.covers("mgmt.tag.list.happy_path")
    def test_tag_list_reports_created_tag(self, client: ManagementClient, resources: ResourceManager) -> None:
        name = f"e2e-mgmt-tag-{unique_marker()}"
        description = "coverage: tag inventory"
        assert all(entry.name != name for entry in _tag_list(client)), (
            f"tag {name!r} was already listed by /tag/list before /tag/new created it"
        )

        _create_tag(client, TagCreateBody(name=name, description=description))
        resources.defer(lambda: _delete_tag(client, name))

        entry = _poll(
            client.proxy,
            lambda: next((entry for entry in _tag_list(client) if entry.name == name), None),
            f"/tag/list never listed {name!r} after /tag/new",
        )
        assert entry.description == description, (
            f"/tag/list reports description {entry.description!r} for {name!r}, configured {description!r}"
        )

    @pytest.mark.covers("mgmt.tag.delete.persists")
    def test_tag_delete_removes_from_list(self, client: ManagementClient, resources: ResourceManager) -> None:
        """The teardown's deferred delete fires again on the already-deleted tag by
        design: it is the safety net if this test fails before the in-body delete,
        and a repeat /tag/delete is a warn-only no-op the teardown absorbs."""
        name = f"e2e-mgmt-tag-{unique_marker()}"
        _create_tag(client, TagCreateBody(name=name))
        resources.defer(lambda: _delete_tag(client, name))

        _ = _poll(
            client.proxy,
            lambda: True if any(entry.name == name for entry in _tag_list(client)) else None,
            f"/tag/list never listed {name!r} after /tag/new; cannot prove deletion removes it",
        )

        _delete_tag_strict(client, name)

        _ = _poll(
            client.proxy,
            lambda: True if all(entry.name != name for entry in _tag_list(client)) else None,
            f"{name!r} still present in /tag/list after /tag/delete at the deadline",
        )


class TestModelAccessGroupRoutes:
    @pytest.mark.covers("mgmt.access_group.new.happy_path")
    def test_new_access_group_tags_the_deployment(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        model_name = f"e2e-mgmt-agmodel-{unique_marker()}"
        _ = _create_db_model(client, resources, model_name)

        access_group = f"e2e-mgmt-ag-{unique_marker()}"
        created = _create_access_group(
            client, AccessGroupNewBody(access_group=access_group, model_names=[model_name])
        )
        resources.defer(lambda: _delete_access_group(client, access_group))

        assert created.access_group == access_group, (
            f"/access_group/new echoed access_group {created.access_group!r}, requested {access_group!r}"
        )
        assert created.models_updated >= 1, (
            f"/access_group/new tagged {created.models_updated} deployments for {model_name!r}, expected >= 1"
        )

        info = _poll(
            client.proxy,
            lambda: _access_group_info(client, access_group),
            f"/access_group/{access_group}/info never resolved the group created by /access_group/new",
        )
        assert model_name in info.model_names, (
            f"the group created by /access_group/new does not list {model_name!r} on read-back; "
            f"/access_group/info reports members {info.model_names}"
        )

    @pytest.mark.covers("mgmt.access_group.info.happy_path")
    def test_access_group_info_reports_membership(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        model_name = f"e2e-mgmt-agmodel-{unique_marker()}"
        _ = _create_db_model(client, resources, model_name)

        access_group = f"e2e-mgmt-ag-{unique_marker()}"
        _ = _create_access_group(
            client, AccessGroupNewBody(access_group=access_group, model_names=[model_name])
        )
        resources.defer(lambda: _delete_access_group(client, access_group))

        info = _poll(
            client.proxy,
            lambda: _access_group_info(client, access_group),
            f"/access_group/{access_group}/info never resolved the created access group",
        )
        assert info.access_group == access_group, (
            f"/access_group/info reports access_group {info.access_group!r}, created {access_group!r}"
        )
        assert model_name in info.model_names, (
            f"/access_group/info reports members {info.model_names}, expected to include {model_name!r}"
        )
        assert info.deployment_count >= 1, (
            f"/access_group/info reports deployment_count {info.deployment_count}, expected >= 1"
        )
