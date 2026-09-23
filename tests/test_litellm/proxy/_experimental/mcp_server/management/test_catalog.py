import pytest

from litellm.proxy._experimental.mcp_server.management.catalog import (
    MANAGEMENT_TOOLS,
    MANAGEMENT_TOOLS_BY_NAME,
    mcp_tools,
)


def test_catalog_lists_exactly_ten_tools_in_order():
    assert [t.name for t in mcp_tools()] == [
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
    ]
    assert set(MANAGEMENT_TOOLS_BY_NAME) == {t.name for t in MANAGEMENT_TOOLS}


def test_every_tool_declares_proxy_admin_and_effect():
    for tool in mcp_tools():
        assert "proxy-admin" in (tool.description or ""), tool.name


def test_tool_annotations_match_mutability():
    by_name = {t.name: t for t in mcp_tools()}
    assert by_name["list_virtual_keys"].annotations.read_only_hint is True
    assert by_name["get_virtual_key"].annotations.read_only_hint is True
    assert by_name["list_access_groups"].annotations.read_only_hint is True
    assert by_name["get_access_group"].annotations.read_only_hint is True
    assert by_name["create_virtual_key"].annotations.read_only_hint is False
    assert by_name["update_virtual_key"].annotations.read_only_hint is False
    assert by_name["delete_virtual_keys"].annotations.destructive_hint is True
    assert by_name["delete_access_group"].annotations.destructive_hint is True
    assert by_name["create_virtual_key"].annotations.destructive_hint is False
    assert by_name["update_access_group"].annotations.idempotent_hint is True
    assert by_name["create_access_group"].annotations.idempotent_hint is False


def test_update_access_group_schema_keeps_nested_defs_and_required_fields():
    schema = MANAGEMENT_TOOLS_BY_NAME["update_access_group"].arguments_model.model_json_schema()
    assert "access_group_id" in schema["properties"]
    assert "data" in schema["properties"]
    assert set(schema["required"]) == {"access_group_id", "data"}
    data_ref = schema["properties"]["data"]
    ref_name = data_ref["$ref"].rsplit("/", 1)[-1]
    nested = schema["$defs"][ref_name]
    assert "access_group_name" in nested["properties"]
    # no required fields: every update field is optional (merge patch)
    assert "required" not in nested


def test_update_virtual_key_schema_is_self_contained():
    schema = MANAGEMENT_TOOLS_BY_NAME["update_virtual_key"].arguments_model.model_json_schema()
    assert "$defs" in schema
    # UpdateKeyRequest has no hard-required fields at schema level (validator
    # requires key or key_alias), but key/key_alias must be present
    assert "key" in schema["properties"]
    assert "key_alias" in schema["properties"]


def test_list_virtual_keys_schema_mirrors_list_keys_params():
    schema = MANAGEMENT_TOOLS_BY_NAME["list_virtual_keys"].arguments_model.model_json_schema()
    props = schema["properties"]
    for name in (
        "page",
        "size",
        "user_id",
        "team_id",
        "organization_id",
        "key_hash",
        "key_alias",
        "search",
        "return_full_object",
        "include_team_keys",
        "include_created_by_keys",
        "sort_by",
        "sort_order",
        "expand",
        "status",
        "project_id",
        "access_group_id",
        "agent_id",
        "substring_matching",
        "expires",
    ):
        assert name in props, name
    assert props["page"]["default"] == 1
    assert props["page"]["minimum"] == 1
    assert props["size"]["maximum"] == 100
    assert props["sort_order"]["default"] == "desc"


def test_list_access_groups_accepts_no_arguments():
    schema = MANAGEMENT_TOOLS_BY_NAME["list_access_groups"].arguments_model.model_json_schema()
    assert schema["properties"] == {}
    MANAGEMENT_TOOLS_BY_NAME["list_access_groups"].arguments_model.model_validate({})
