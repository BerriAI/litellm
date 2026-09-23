"""Management MCP endpoint e2e: the opt-in built-in MCP server at
/litellm-management/mcp on the control plane, driven by the official mcp SDK
streamable-http client in management_mcp_client.py.

Covers the lifecycle contract through the caller-visible surface: tool listing,
virtual-key and access-group round trips with REST read-back, admin-only
admission (401/403 on initialize), allowed_routes gating, argument validation
errors, and per-request caller isolation when the same mcp-session-id header is
reused.
"""

import re
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import NoBody, Result, Success, unwrap
from lifecycle import ResourceManager
from management.management_mcp_client import (
    call_tool,
    call_tool_as,
    initialize_status,
    list_tool_names,
    management_mcp,
    run,
)
from management_client import ManagementClient
from models import KeyGenerateBody, KeyInfoResponse, KeyInfoParams, KeyUpdateBody, UserNewBody
from pydantic import BaseModel

EXPECTED_TOOLS: Final = (
    "list_virtual_keys",
    "get_virtual_key",
    "create_virtual_key",
    "update_virtual_key",
    "delete_virtual_keys",
    "list_access_groups",
    "get_access_group",
    "create_access_group",
    "update_access_group",
    "delete_access_group",
)


class _UserNewResult(BaseModel):
    user_id: str
    key: str


class _CreatedKey(BaseModel):
    key: str


class _AccessGroupResult(BaseModel):
    access_group_id: str
    access_group_name: str
    description: str | None = None


class _AccessGroupList(BaseModel):
    result: list[_AccessGroupResult]


class _KeyListEntry(BaseModel):
    key_alias: str | None = None
    created_by: str | None = None


class _KeyListResult(BaseModel):
    keys: list[_KeyListEntry]
    total_count: int | None = None


def _create_admin_key(client: ManagementClient, resources: ResourceManager) -> tuple[str, str]:
    """A distinct proxy-admin key minted by a dedicated proxy_admin user via
    /user/new auto_create_key, so it carries the role without being the master
    key. Returns (user_id, key)."""
    marker = unique_marker()
    created: Final = unwrap(
        client.proxy.transport.post(
            "/user/new",
            headers=client.proxy.management_headers(),
            json=UserNewBody(
                user_email=f"e2e-mgmt-mcp-{marker}@example.com",
                user_role="proxy_admin",
                auto_create_key=True,
            ),
            response_type=_UserNewResult,
        )
    )
    resources.defer(lambda: client.delete_user(created.user_id))
    return created.user_id, created.key


def _set_allowed_routes(client: ManagementClient, key: str, routes: list[str]) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/key/update",
            headers=client.proxy.management_headers(),
            json=KeyUpdateBody(key=key, allowed_routes=routes),
            response_type=NoBody,
        )
    )


def _key_info(client: ManagementClient, key: str) -> Result[KeyInfoResponse]:
    return client.proxy.transport.get(
        "/key/info",
        headers=client.proxy.management_headers(),
        params=KeyInfoParams(key=key),
        response_type=KeyInfoResponse,
    )


def _access_group_info(client: ManagementClient, access_group_id: str) -> Result[_AccessGroupResult]:
    return client.proxy.transport.get(
        f"/v1/access_group/{access_group_id}",
        headers=client.proxy.management_headers(),
        params=NoBody(),
        response_type=_AccessGroupResult,
    )


def _delete_access_group(client: ManagementClient, access_group_id: str) -> None:
    _ = client.proxy.transport.delete(
        f"/v1/access_group/{access_group_id}",
        headers=client.proxy.management_headers(),
        json=NoBody(),
        response_type=NoBody,
    )


@pytest.mark.e2e
class TestManagementMCPTools:
    def test_list_tools_returns_exactly_the_ten_management_tools(self, client: ManagementClient) -> None:
        names: Final = run(list_tool_names(management_mcp(key=client.master_key)))
        assert sorted(names) == sorted(EXPECTED_TOOLS), f"expected the 10 management tools, got {names!r}"

    def test_virtual_key_round_trip(self, client: ManagementClient, resources: ResourceManager) -> None:
        alias: Final = f"e2e-mgmt-mcp-{unique_marker()}"
        updated_alias: Final = f"{alias}-renamed"
        admin = management_mcp(key=client.master_key)

        created_outcome, created = run(call_tool_as(admin, "create_virtual_key", {"key_alias": alias}, _CreatedKey))
        assert not created_outcome.is_error and created is not None
        assert created.key.startswith("sk-"), f"create_virtual_key must return an sk- key, got {created.key!r}"
        resources.defer(lambda: client.proxy.delete_key(created.key))

        rest_info: Final = _key_info(client, created.key)
        assert isinstance(rest_info, Success), f"REST /key/info must see the key, got {rest_info!r}"
        assert rest_info.data.info.key_alias == alias

        updated: Final = run(call_tool(admin, "update_virtual_key", {"key": created.key, "key_alias": updated_alias}))
        assert not updated.is_error, updated.text
        rest_updated: Final = _key_info(client, created.key)
        assert isinstance(rest_updated, Success) and rest_updated.data.info.key_alias == updated_alias

        info: Final = run(call_tool(admin, "get_virtual_key", {"key": created.key}))
        assert not info.is_error and info.structured is not None
        assert not re.search(r"sk-[A-Za-z0-9_-]+", repr(info.structured)), (
            f"get_virtual_key leaked a raw key: {info.structured!r}"
        )

        _, listed = run(
            call_tool_as(
                admin,
                "list_virtual_keys",
                {"key_alias": updated_alias, "return_full_object": True},
                _KeyListResult,
            )
        )
        assert listed is not None
        aliases: Final = [entry.key_alias for entry in listed.keys]
        assert updated_alias in aliases, f"list_virtual_keys filter missed {updated_alias!r}: {aliases!r}"

        deleted: Final = run(call_tool(admin, "delete_virtual_keys", {"keys": [created.key]}))
        assert not deleted.is_error and deleted.structured is not None
        assert not re.search(r"sk-[A-Za-z0-9_-]+", repr(deleted.structured)), (
            f"delete_virtual_keys echoed a raw key: {deleted.structured!r}"
        )

        after: Final = _key_info(client, created.key)
        assert not isinstance(after, Success) or after.data.info.status == "deleted", (
            f"REST /key/info must show the key deleted, got {after!r}"
        )

    def test_access_group_round_trip(self, client: ManagementClient, resources: ResourceManager) -> None:
        name: Final = f"e2e-mgmt-mcp-ag-{unique_marker()}"
        admin = management_mcp(key=client.master_key)

        created_outcome, created = run(
            call_tool_as(admin, "create_access_group", {"access_group_name": name, "description": "one"}, _AccessGroupResult)
        )
        assert not created_outcome.is_error and created is not None
        group_id: Final = created.access_group_id
        resources.defer(lambda: _delete_access_group(client, group_id))

        _, fetched = run(call_tool_as(admin, "get_access_group", {"access_group_id": group_id}, _AccessGroupResult))
        assert fetched is not None and fetched.access_group_name == name and fetched.description == "one"

        _, listed = run(call_tool_as(admin, "list_access_groups", {}, _AccessGroupList))
        assert listed is not None
        names: Final = [entry.access_group_name for entry in listed.result]
        assert name in names, f"list_access_groups missed {name!r}: {names!r}"

        _, updated = run(
            call_tool_as(
                admin,
                "update_access_group",
                {"access_group_id": group_id, "data": {"description": "two"}},
                _AccessGroupResult,
            )
        )
        assert updated is not None and updated.description == "two"

        rest: Final = _access_group_info(client, group_id)
        assert isinstance(rest, Success) and rest.data.description == "two"

        deleted: Final = run(call_tool(admin, "delete_access_group", {"access_group_id": group_id}))
        assert not deleted.is_error and deleted.structured == {}, (
            f"delete_access_group must return an empty object, got {deleted.structured!r}"
        )
        assert not isinstance(_access_group_info(client, group_id), Success)

    def test_non_admin_key_denied_and_trailing_slash_and_no_auth(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        non_admin: Final = client.proxy.generate_key(
            KeyGenerateBody(models=[], user_id=f"e2e-{unique_marker()}")
        )
        resources.defer(lambda: client.proxy.delete_key(non_admin))
        status: Final = initialize_status(management_mcp(key=non_admin))
        assert status in (401, 403), f"non-admin initialize must be denied, got {status}"

        names: Final = run(list_tool_names(management_mcp(path="/litellm-management/mcp/", key=client.master_key)))
        assert len(names) == 10

        assert initialize_status(management_mcp()) == 401

    def test_allowed_routes_gating(self, client: ManagementClient, resources: ResourceManager) -> None:
        _, mcp_only_key = _create_admin_key(client, resources)
        _set_allowed_routes(client, mcp_only_key, ["mcp_routes"])
        status: Final = initialize_status(management_mcp(key=mcp_only_key))
        assert status == 403, f"mcp_routes-only key must be denied at admission, got {status}"

        _, management_key = _create_admin_key(client, resources)
        _set_allowed_routes(client, management_key, ["management_routes"])
        names: Final = run(list_tool_names(management_mcp(key=management_key)))
        assert len(names) == 10, f"management_routes key must list tools, got {names!r}"

    def test_invalid_arguments_return_iserror_without_input_echo(self, client: ManagementClient) -> None:
        outcome: Final = run(
            call_tool(
                management_mcp(key=client.master_key),
                "list_virtual_keys",
                {"size": 0, "key_alias": "sk-should-not-echo"},
            )
        )
        assert outcome.is_error, f"size=0 must produce an isError result, got {outcome.text!r}"
        blob: Final = outcome.text + repr(outcome.structured)
        assert "sk-should-not-echo" not in blob, f"validation detail echoed the caller input: {blob[:400]}"
        assert "size" in blob

    def test_reused_session_id_isolates_callers(self, client: ManagementClient, resources: ResourceManager) -> None:
        user_a, key_a = _create_admin_key(client, resources)
        user_b, key_b = _create_admin_key(client, resources)
        shared_session: Final = f"e2e-shared-{unique_marker()}"
        marker_a: Final = f"e2e-mcp-iso-a-{unique_marker()}"
        marker_b: Final = f"e2e-mcp-iso-b-{unique_marker()}"
        session_a = management_mcp(key=key_a, session_id=shared_session)
        session_b = management_mcp(key=key_b, session_id=shared_session)

        _, created_a = run(call_tool_as(session_a, "create_virtual_key", {"key_alias": marker_a}, _CreatedKey))
        _, created_b = run(call_tool_as(session_b, "create_virtual_key", {"key_alias": marker_b}, _CreatedKey))
        assert created_a is not None and created_b is not None
        resources.defer(lambda: client.proxy.delete_key(created_a.key))
        resources.defer(lambda: client.proxy.delete_key(created_b.key))

        _, listed_a = run(
            call_tool_as(
                session_a,
                "list_virtual_keys",
                {"key_alias": marker_a, "return_full_object": True},
                _KeyListResult,
            )
        )
        assert listed_a is not None and listed_a.keys
        assert all(entry.created_by == user_a for entry in listed_a.keys), (
            f"caller A must only see keys created_by {user_a!r}, got {listed_a.keys!r}"
        )
        assert listed_a.keys[0].key_alias == marker_a

        _, listed_b = run(
            call_tool_as(
                session_b,
                "list_virtual_keys",
                {"key_alias": marker_b, "return_full_object": True},
                _KeyListResult,
            )
        )
        assert listed_b is not None and listed_b.keys
        assert all(entry.created_by == user_b for entry in listed_b.keys), (
            f"caller B must only see keys created_by {user_b!r}, got {listed_b.keys!r}"
        )
        assert listed_b.keys[0].key_alias == marker_b
