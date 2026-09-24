"""Management MCP endpoint e2e: the opt-in built-in MCP server at
/litellm-management/mcp on the control plane, driven by the official mcp SDK
streamable-http client in management_mcp_client.py.

The catalog is generated from the proxy's own OpenAPI spec, so tools carry the
real operation names and the {path, query, body} argument shape. Covers
initialize/list, existing-role admission, allowed_routes gating, and one live
round trip per family (key, team, user, model, access group, budget) with REST
read-back where the spec exposes no matching read tool.
"""

import json
import re
from typing import Final, cast

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
from models import (
    KeyDeleteBody,
    KeyInfoParams,
    KeyInfoResponse,
    KeyUpdateBody,
    LiteLLMParamsBody,
    TeamDeleteBody,
    UserNewBody,
    UserRole,
)
from pydantic import BaseModel


class _UserNewResult(BaseModel):
    user_id: str
    key: str


class _CreatedKey(BaseModel):
    key: str
    key_alias: str | None = None


class _TeamResult(BaseModel):
    team_id: str
    team_alias: str | None = None


class _UserResult(BaseModel):
    user_id: str
    user_email: str | None = None


class _ModelResult(BaseModel):
    model_id: str | None = None
    model_info: dict[str, object] | None = None


class _AccessGroupResult(BaseModel):
    access_group_id: str
    access_group_name: str
    description: str | None = None


class _AccessGroupList(BaseModel):
    result: list[_AccessGroupResult]


class _BudgetDeleteBody(BaseModel):
    id: str


def _create_user_key(
    client: ManagementClient, resources: ResourceManager, role: UserRole = "proxy_admin"
) -> tuple[str, str]:
    marker = unique_marker()
    created: Final = unwrap(
        client.proxy.transport.post(
            "/user/new",
            headers=client.proxy.management_headers(),
            json=UserNewBody(
                user_email=f"e2e-mgmt-mcp-{marker}@example.com",
                user_role=role,
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
    def test_list_tools_returns_the_openapi_generated_catalog(self, client: ManagementClient) -> None:
        names: Final = run(list_tool_names(management_mcp(key=client.master_key)))
        assert len(names) > 100, f"expected the OpenAPI-generated catalog, got {len(names)} tools"
        for required in (
            "generate_key_fn_key_generate_post",
            "new_team_team_new_post",
            "new_user_user_new_post",
            "add_new_model_model_new_post",
            "create_access_group_v1_access_group_post",
            "new_budget_budget_new_post",
        ):
            assert required in names, f"catalog missing {required!r}"
        assert "list_virtual_keys" not in names, "the handwritten ten-tool catalog must be gone"

    def test_virtual_key_round_trip(self, client: ManagementClient, resources: ResourceManager) -> None:
        alias: Final = f"e2e-mgmt-mcp-{unique_marker()}"
        updated_alias: Final = f"{alias}-renamed"
        admin = management_mcp(key=client.master_key)

        created_outcome, created = run(
            call_tool_as(admin, "generate_key_fn_key_generate_post", {"body": {"key_alias": alias}}, _CreatedKey)
        )
        assert not created_outcome.is_error and created is not None
        assert created.key.startswith("sk-"), f"key generate must return an sk- key, got {created.key!r}"
        resources.defer(lambda: client.proxy.delete_key(created.key))

        rest_info: Final = _key_info(client, created.key)
        assert isinstance(rest_info, Success), f"REST /key/info must see the key, got {rest_info!r}"
        assert rest_info.data.info.key_alias == alias

        updated: Final = run(
            call_tool(
                admin,
                "update_key_fn_key_update_post",
                {"body": {"key": created.key, "key_alias": updated_alias}},
            )
        )
        assert not updated.is_error, updated.text
        rest_updated: Final = _key_info(client, created.key)
        assert isinstance(rest_updated, Success) and rest_updated.data.info.key_alias == updated_alias

        deleted: Final = run(call_tool(admin, "delete_key_fn_key_delete_post", {"body": {"keys": [created.key]}}))
        assert not deleted.is_error, deleted.text

        after: Final = _key_info(client, created.key)
        assert not isinstance(after, Success) or after.data.info.status == "deleted", (
            f"REST /key/info must show the key deleted, got {after!r}"
        )

    def test_team_round_trip(self, client: ManagementClient, resources: ResourceManager) -> None:
        alias: Final = f"e2e-mgmt-mcp-team-{unique_marker()}"
        admin = management_mcp(key=client.master_key)

        created_outcome, created = run(
            call_tool_as(admin, "new_team_team_new_post", {"body": {"team_alias": alias}}, _TeamResult)
        )
        assert not created_outcome.is_error and created is not None
        team_id: Final = created.team_id
        resources.defer(
            lambda: client.proxy.transport.post(
                "/team/delete",
                headers=client.proxy.management_headers(),
                json=TeamDeleteBody(team_ids=[team_id]),
                response_type=NoBody,
            )
        )
        rest_team: Final = client.team_info(team_id)
        assert rest_team.team_alias == alias

        updated: Final = run(
            call_tool(
                admin,
                "update_team_team_update_post",
                {"body": {"team_id": team_id, "team_alias": f"{alias}-2"}},
            )
        )
        assert not updated.is_error, updated.text
        assert client.team_info(team_id).team_alias == f"{alias}-2"

        deleted: Final = run(call_tool(admin, "delete_team_team_delete_post", {"body": {"team_ids": [team_id]}}))
        assert not deleted.is_error, deleted.text
        assert client.team_info_status(team_id).status_code != 200

    def test_user_round_trip(self, client: ManagementClient, resources: ResourceManager) -> None:
        email: Final = f"e2e-mgmt-mcp-user-{unique_marker()}@example.com"
        admin = management_mcp(key=client.master_key)

        created_outcome, created = run(
            call_tool_as(admin, "new_user_user_new_post", {"body": {"user_email": email}}, _UserResult)
        )
        assert not created_outcome.is_error and created is not None
        user_id: Final = created.user_id
        resources.defer(lambda: client.delete_user(user_id))
        assert client.user_info(user_id).user_info.user_email == email

        updated_email: Final = f"renamed-{email}"
        updated: Final = run(
            call_tool(
                admin,
                "user_update_user_update_post",
                {"body": {"user_id": user_id, "user_email": updated_email}},
            )
        )
        assert not updated.is_error, updated.text
        assert client.user_info(user_id).user_info.user_email == updated_email

        deleted: Final = run(call_tool(admin, "delete_user_user_delete_post", {"body": {"user_ids": [user_id]}}))
        assert not deleted.is_error, deleted.text

    def test_model_round_trip(self, client: ManagementClient, resources: ResourceManager) -> None:
        model_name: Final = f"e2e-mgmt-mcp-model-{unique_marker()}"
        admin = management_mcp(key=client.master_key)

        created_outcome, created = run(
            call_tool_as(
                admin,
                "add_new_model_model_new_post",
                {
                    "body": {
                        "model_name": model_name,
                        "litellm_params": LiteLLMParamsBody(
                            model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"
                        ).model_dump(exclude_none=True),
                    }
                },
                _ModelResult,
            )
        )
        assert not created_outcome.is_error and created is not None, created_outcome.text
        model_id: Final = str((created.model_info or {}).get("id") or created.model_id)
        assert model_id and model_id != "None", f"/model/new must return a model id, got {created!r}"
        resources.defer(lambda: client.proxy.delete_model(model_id))
        assert any(entry.model_name == model_name for entry in client.proxy.model_info()), (
            f"REST /model/info must list {model_name!r}"
        )

        updated: Final = run(
            call_tool(
                admin,
                "update_model_model_update_post",
                {
                    "body": {
                        "model_info": {"id": model_id},
                        "model_name": f"{model_name}-v2",
                        "litellm_params": LiteLLMParamsBody(
                            model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"
                        ).model_dump(exclude_none=True),
                    }
                },
            )
        )
        assert not updated.is_error, updated.text
        assert any(entry.model_name == f"{model_name}-v2" for entry in client.proxy.model_info()), (
            "REST /model/info must reflect the updated model_name"
        )

        deleted: Final = run(call_tool(admin, "delete_model_model_delete_post", {"body": {"id": model_id}}))
        assert not deleted.is_error, deleted.text
        assert all(entry.model_name != f"{model_name}-v2" for entry in client.proxy.model_info())

    def test_access_group_round_trip(self, client: ManagementClient, resources: ResourceManager) -> None:
        name: Final = f"e2e-mgmt-mcp-ag-{unique_marker()}"
        admin = management_mcp(key=client.master_key)

        created_outcome, created = run(
            call_tool_as(
                admin,
                "create_access_group_v1_access_group_post",
                {"body": {"access_group_name": name, "description": "one"}},
                _AccessGroupResult,
            )
        )
        assert not created_outcome.is_error and created is not None
        group_id: Final = created.access_group_id
        resources.defer(lambda: _delete_access_group(client, group_id))

        _, fetched = run(
            call_tool_as(
                admin,
                "get_access_group_v1_access_group__access_group_id__get",
                {"path": {"access_group_id": group_id}},
                _AccessGroupResult,
            )
        )
        assert fetched is not None and fetched.access_group_name == name and fetched.description == "one"

        listed: Final = run(call_tool(admin, "list_access_groups_v1_access_group_get", {}))
        assert not listed.is_error, listed.text
        listed_groups: Final = _AccessGroupList(
            result=[_AccessGroupResult.model_validate(entry) for entry in cast(list[object], json.loads(listed.text))]
        )
        assert name in [entry.access_group_name for entry in listed_groups.result]

        _, updated = run(
            call_tool_as(
                admin,
                "update_access_group_v1_access_group__access_group_id__put",
                {"path": {"access_group_id": group_id}, "body": {"description": "two"}},
                _AccessGroupResult,
            )
        )
        assert updated is not None and updated.description == "two"
        rest: Final = _access_group_info(client, group_id)
        assert isinstance(rest, Success) and rest.data.description == "two"

        deleted: Final = run(
            call_tool(
                admin,
                "delete_access_group_v1_access_group__access_group_id__delete",
                {"path": {"access_group_id": group_id}},
            )
        )
        assert not deleted.is_error, deleted.text
        assert not isinstance(_access_group_info(client, group_id), Success)

    def test_budget_round_trip(self, client: ManagementClient, resources: ResourceManager) -> None:
        budget_id: Final = f"e2e-mgmt-mcp-budget-{unique_marker()}"
        admin = management_mcp(key=client.master_key)

        created: Final = run(
            call_tool(
                admin,
                "new_budget_budget_new_post",
                {"body": {"budget_id": budget_id, "max_budget": 5.0}},
            )
        )
        assert not created.is_error, created.text
        resources.defer(
            lambda: client.proxy.transport.post(
                "/budget/delete",
                headers=client.proxy.management_headers(),
                json=_BudgetDeleteBody(id=budget_id),
                response_type=NoBody,
            )
        )

        info: Final = run(call_tool(admin, "info_budget_budget_info_post", {"body": {"budgets": [budget_id]}}))
        assert not info.is_error, info.text
        assert budget_id in info.text, f"/budget/info via the tool must name {budget_id!r}: {info.text[:300]}"

        deleted: Final = run(call_tool(admin, "delete_budget_budget_delete_post", {"body": {"id": budget_id}}))
        assert not deleted.is_error, deleted.text

    def test_non_admin_permissions_and_trailing_slash_and_no_auth(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        _, non_admin = _create_user_key(client, resources, role="internal_user")
        status: Final = initialize_status(management_mcp(key=non_admin))
        assert status == 200, f"authenticated non-admin initialize must succeed, got {status}"

        own_key: Final = run(
            call_tool(management_mcp(key=non_admin), "info_key_fn_key_info_get", {"query": {"key": non_admin}})
        )
        assert not own_key.is_error, own_key.text
        rest_own_key: Final = client.proxy.transport.get(
            "/key/info",
            headers=client.proxy.management_headers(caller_key=non_admin),
            params=KeyInfoParams(key=non_admin),
            response_type=KeyInfoResponse,
        )
        assert isinstance(rest_own_key, Success)

        _, other_key = _create_user_key(client, resources, role="internal_user")
        denied: Final = run(
            call_tool(management_mcp(key=non_admin), "delete_key_fn_key_delete_post", {"body": {"keys": [other_key]}})
        )
        assert denied.is_error, "one user's key must not delete another user's key"
        rest_denied: Final = client.proxy.transport.post(
            "/key/delete",
            headers=client.proxy.management_headers(caller_key=non_admin),
            json=KeyDeleteBody(keys=[other_key]),
            response_type=NoBody,
        )
        assert not isinstance(rest_denied, Success)
        surviving: Final = _key_info(client, other_key)
        assert isinstance(surviving, Success) and surviving.data.info.status != "deleted"

        names: Final = run(list_tool_names(management_mcp(path="/litellm-management/mcp/", key=client.master_key)))
        assert len(names) > 100

        assert initialize_status(management_mcp()) == 401

    def test_allowed_routes_gating(self, client: ManagementClient, resources: ResourceManager) -> None:
        _, mcp_only_key = _create_user_key(client, resources)
        _set_allowed_routes(client, mcp_only_key, ["mcp_routes"])
        status: Final = initialize_status(management_mcp(key=mcp_only_key))
        assert status == 403, f"mcp_routes-only key must be denied at admission, got {status}"

        _, management_key = _create_user_key(client, resources)
        _set_allowed_routes(client, management_key, ["management_routes"])
        names: Final = run(list_tool_names(management_mcp(key=management_key)))
        assert len(names) > 100, f"management_routes key must list tools, got {names!r}"

        _set_allowed_routes(client, management_key, ["management_mcp_routes", "/key/info"])
        allowed: Final = run(
            call_tool(
                management_mcp(key=management_key), "info_key_fn_key_info_get", {"query": {"key": management_key}}
            )
        )
        assert not allowed.is_error, allowed.text
        denied: Final = run(
            call_tool(management_mcp(key=management_key), "generate_key_fn_key_generate_post", {"body": {}})
        )
        assert denied.is_error, "MCP entrance permission must not grant target-route permission"

    def test_invalid_arguments_return_iserror_without_input_echo(self, client: ManagementClient) -> None:
        outcome: Final = run(
            call_tool(
                management_mcp(key=client.master_key),
                "generate_key_fn_key_generate_post",
                {"body": {"key_alias": "sk-should-not-echo"}, "headers": {"x": "y"}},
            )
        )
        assert outcome.is_error, f"an unknown argument section must fail, got {outcome.text!r}"
        assert "sk-should-not-echo" not in outcome.text, f"the error must not echo input: {outcome.text[:400]}"

    def test_rest_error_surfaces_verbatim(self, client: ManagementClient) -> None:
        outcome: Final = run(
            call_tool(
                management_mcp(key=client.master_key),
                "update_key_fn_key_update_post",
                {"body": {"key": "sk-does-not-exist", "key_alias": "x"}},
            )
        )
        assert outcome.is_error, f"a missing key must surface the REST error, got {outcome.text!r}"
        assert re.search(r'"error"|not found|404', outcome.text, re.IGNORECASE), outcome.text[:300]
