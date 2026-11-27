"""The auth cache entry the Rust auth store shapes for a team, against Python's own model and its own team reader and
writer. The row and entry literals are the ones ``litellm/rust/crates/auth-store/tests/team.rs`` reads from Postgres
and ``litellm/rust/crates/redis-store/tests/auth_cache_team.rs`` round-trips through Redis. Change all three
together."""

import json
import re
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTableCachedObj
from litellm.proxy.auth.auth_checks import cache_team_object, get_team_object
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

PERMISSION_ROW: Final = (
    r'{"object_permission_id":"permission","mcp_servers":[],"mcp_access_groups":[],'
    r'"mcp_tool_permissions":null,"vector_stores":[],"agents":[],"agent_access_groups":[],'
    r'"models":[],"blocked_tools":[],"mcp_toolsets":[],"search_tools":[],'
    r'"mcp_tool_search_enabled":null,"skills":[]}'
)
CREATED_ROW: Final = (
    r'{"team_id":"created","team_alias":null,"organization_id":null,"admins":[],'
    r'"members":[],"members_with_roles":{},"metadata":{},"max_budget":null,'
    r'"soft_budget":null,"spend":0.0,"models":[],"max_parallel_requests":null,'
    r'"tpm_limit":null,"rpm_limit":null,"tpd_limit":null,"budget_duration":null,'
    r'"budget_reset_at":null,"blocked":false,"created_at":"2026-11-19T10:00:00.123+00:00",'
    r'"updated_at":"2026-11-19T10:00:00+00:00","model_spend":{},"model_max_budget":{},'
    r'"router_settings":{},"team_member_permissions":[],"access_group_ids":[],"policies":[],'
    r'"default_team_member_models":[],"budget_limits":null,"model_id":null,'
    r'"allow_team_guardrail_config":false,"object_permission_id":null,'
    r'"litellm_model_table":null}'
)
CREATED_ENTRY: Final = (
    r'{"team_alias": null, "team_id": "created", "organization_id": null, "admins": [],'
    r' "members": [], "members_with_roles": [], "team_member_permissions": [],'
    r' "metadata": {}, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,'
    r' "max_budget": null, "soft_budget": null, "budget_duration": null,'
    r' "budget_limits": null, "models": [], "blocked": false, "router_settings": {},'
    r' "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,'
    r' "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,'
    r' "model_spend": {}, "model_max_budget": {}, "policies": [],'
    r' "allow_team_guardrail_config": false, "litellm_model_table": null,'
    r' "object_permission": null, "object_permission_id": null,'
    r' "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z",'
    r' "last_refreshed_at": null}'
)
DEFAULTS_ROW: Final = (
    r'{"team_id":"defaults","team_alias":null,"organization_id":"org","admins":[],"members":[],"members_wi'
    r'th_roles":[],"metadata":{},"max_budget":10.0,"soft_budget":null,"spend":0.0,"models":[],"max_paralle'
    r'l_requests":null,"tpm_limit":100,"rpm_limit":10,"tpd_limit":null,"budget_duration":"30d","budget_res'
    r'et_at":"2026-12-01T00:00:00+00:00","blocked":false,"created_at":"2026-11-27T08:00:00+00:00","updated'
    r'_at":"2026-11-27T08:00:00+00:00","model_spend":{},"model_max_budget":{},"router_settings":{},"team_m'
    r'ember_permissions":["/key/generate"],"access_group_ids":["wanted"],"policies":[],"default_team_membe'
    r'r_models":[],"budget_limits":null,"model_id":null,"allow_team_guardrail_config":false,"object_permis'
    r'sion_id":null,"litellm_model_table":null}'
)
DEFAULTS_ENTRY: Final = (
    r'{"team_alias": null, "team_id": "defaults", "organization_id": "org", "admins": [], "members": [], "'
    r'members_with_roles": [], "team_member_permissions": ["/key/generate"], "metadata": {}, "tpm_limit": '
    r'100, "rpm_limit": 10, "tpd_limit": null, "max_budget": 10.0, "soft_budget": null, "budget_duration":'
    r' "30d", "budget_limits": null, "models": [], "blocked": false, "router_settings": {}, "access_group_'
    r'ids": ["wanted"], "default_team_member_models": [], "spend": 0.0, "max_parallel_requests": null, "bu'
    r'dget_reset_at": "2026-12-01T00:00:00Z", "model_id": null, "model_spend": {}, "model_max_budget": {},'
    r' "policies": [], "allow_team_guardrail_config": false, "litellm_model_table": null, "object_permissi'
    r'on": null, "object_permission_id": null, "updated_at": "2026-11-27T08:00:00Z", "created_at": "2026-1'
    r'1-27T08:00:00Z", "last_refreshed_at": null}'
)
FULL_ROW: Final = (
    r'{"team_id":"full","team_alias":"Full","organization_id":"org","admins":["admin"],'
    r'"members":["member"],"members_with_roles":[{"role": "admin", "extra": 1,'
    r' "user_id": "admin"}, {"role": "user", "user_email": "m@example.com"}],'
    r'"metadata":{"guardrails": ["pii"]},"max_budget":10.5,"soft_budget":5.25,"spend":1.5,'
    r'"models":["claude","all-proxy-models"],"max_parallel_requests":3,"tpm_limit":1000,'
    r'"rpm_limit":10,"tpd_limit":100000,"budget_duration":"30d",'
    r'"budget_reset_at":"2026-12-01T00:00:00+00:00","blocked":true,'
    r'"created_at":"2026-11-19T10:00:00.123+00:00","updated_at":"2026-11-19T10:00:00+00:00",'
    r'"model_spend":{"claude": 1.5},"model_max_budget":{"claude": {"budget_limit": 2.0}},'
    r'"router_settings":{"num_retries": 2},"team_member_permissions":["/key/generate"],'
    r'"access_group_ids":["group"],"policies":["policy"],'
    r'"default_team_member_models":["claude"],"budget_limits":[{"reset_at":'
    r' "2026-12-01T00:00:00+00:00", "max_budget": "1", "window_start":'
    r' "2026-11-01T00:00:00+00:00", "budget_duration": "1d"}, {"later": 1,'
    r' "max_budget": 2.5, "budget_duration": "30d"}],"model_id":7,'
    r'"allow_team_guardrail_config":true,"object_permission_id":"permission",'
    r'"litellm_model_table":{"id": 7, "created_by": "test", "updated_by": "test",'
    r' "model_aliases": {"gpt": "claude"}}}'
)
FULL_ENTRY: Final = (
    r'{"team_alias": "Full", "team_id": "full", "organization_id": "org",'
    r' "admins": ["admin"], "members": ["member"],'
    r' "members_with_roles": [{"user_id": "admin", "user_email": null, "role": "admin"},'
    r' {"user_id": null, "user_email": "m@example.com", "role": "user"}],'
    r' "team_member_permissions": ["/key/generate"], "metadata": {"guardrails": ["pii"]},'
    r' "tpm_limit": 1000, "rpm_limit": 10, "tpd_limit": 100000, "max_budget": 10.5,'
    r' "soft_budget": 5.25, "budget_duration": "30d",'
    r' "budget_limits": [{"budget_duration": "1d", "max_budget": 1.0,'
    r' "reset_at": "2026-12-01T00:00:00Z", "window_start": "2026-11-01T00:00:00Z"},'
    r' {"budget_duration": "30d", "max_budget": 2.5, "reset_at": null}],'
    r' "models": ["claude", "all-proxy-models"],'
    r' "blocked": true, "router_settings": {"num_retries": 2},'
    r' "access_group_ids": ["group"], "default_team_member_models": ["claude"],'
    r' "spend": 1.5, "max_parallel_requests": 3, "budget_reset_at": "2026-12-01T00:00:00Z",'
    r' "model_id": 7, "model_spend": {"claude": 1.5},'
    r' "model_max_budget": {"claude": {"budget_limit": 2.0}}, "policies": ["policy"],'
    r' "allow_team_guardrail_config": true, "litellm_model_table": {"id": 7,'
    r' "model_aliases": {"gpt": "claude"}, "created_by": "test", "updated_by": "test",'
    r' "team": null}, "object_permission": {"object_permission_id": "permission",'
    r' "mcp_servers": [], "mcp_access_groups": [], "mcp_tool_permissions": null,'
    r' "vector_stores": [], "agents": [], "agent_access_groups": [], "models": [],'
    r' "mcp_toolsets": [], "blocked_tools": [], "search_tools": [],'
    r' "mcp_tool_search_enabled": null, "skills": [], "apps": null},'
    r' "object_permission_id": "permission",'
    r' "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z",'
    r' "last_refreshed_at": null}'
)

STAMPED_ENTRY: Final = (
    r'{"team_alias": null, "team_id": "stamped", "organization_id": null, "admins": [],'
    r' "members": [], "members_with_roles": [], "team_member_permissions": [],'
    r' "metadata": {}, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,'
    r' "max_budget": null, "soft_budget": null, "budget_duration": null,'
    r' "budget_limits": null, "models": [], "blocked": false, "router_settings": {},'
    r' "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,'
    r' "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,'
    r' "model_spend": {}, "model_max_budget": {}, "policies": [],'
    r' "allow_team_guardrail_config": false, "litellm_model_table": null,'
    r' "object_permission": null, "object_permission_id": null,'
    r' "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z",'
    r' "last_refreshed_at": 1764547200.5}'
)

NULLS_ROW: Final = (
    r'{"team_id":"nulls","team_alias":null,"organization_id":null,"admins":[],"members":[],'
    r'"members_with_roles":{},"metadata":null,"max_budget":null,"soft_budget":null,'
    r'"spend":0.0,"models":[],"max_parallel_requests":null,"tpm_limit":null,"rpm_limit":null,'
    r'"tpd_limit":null,"budget_duration":null,"budget_reset_at":null,"blocked":false,'
    r'"created_at":"2026-11-19T10:00:00+00:00","updated_at":"2026-11-19T10:00:00+00:00",'
    r'"model_spend":null,"model_max_budget":null,"router_settings":{},'
    r'"team_member_permissions":[],"access_group_ids":[],"policies":[],'
    r'"default_team_member_models":[],"budget_limits":null,"model_id":null,'
    r'"allow_team_guardrail_config":false,"object_permission_id":null,'
    r'"litellm_model_table":null}'
)
NULLS_ENTRY: Final = (
    r'{"team_alias": null, "team_id": "nulls", "organization_id": null, "admins": [],'
    r' "members": [], "members_with_roles": [], "team_member_permissions": [],'
    r' "metadata": null, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,'
    r' "max_budget": null, "soft_budget": null, "budget_duration": null,'
    r' "budget_limits": null, "models": [], "blocked": false, "router_settings": {},'
    r' "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,'
    r' "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,'
    r' "model_spend": null, "model_max_budget": null, "policies": [],'
    r' "allow_team_guardrail_config": false, "litellm_model_table": null,'
    r' "object_permission": null, "object_permission_id": null,'
    r' "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00Z",'
    r' "last_refreshed_at": null}'
)
TEXT_ROW: Final = (
    r'{"team_id":"text","team_alias":null,"organization_id":null,"admins":[],"members":[],'
    r'"members_with_roles":{},"metadata":"{\"guardrails\": [\"pii\"]}","max_budget":null,'
    r'"soft_budget":null,"spend":0.0,"models":[],"max_parallel_requests":null,'
    r'"tpm_limit":null,"rpm_limit":null,"tpd_limit":null,"budget_duration":null,'
    r'"budget_reset_at":null,"blocked":false,"created_at":"2026-11-19T10:00:00+00:00",'
    r'"updated_at":"2026-11-19T10:00:00+00:00","model_spend":{},'
    r'"model_max_budget":"{\"claude\": {\"budget_limit\": 2.0}}",'
    r'"router_settings":"{\"num_retries\": 2}","team_member_permissions":[],'
    r'"access_group_ids":[],"policies":[],"default_team_member_models":[],'
    r'"budget_limits":"[{\"budget_duration\": \"1d\", \"max_budget\": 1}]","model_id":null,'
    r'"allow_team_guardrail_config":false,"object_permission_id":null,'
    r'"litellm_model_table":null}'
)
TEXT_ENTRY: Final = (
    r'{"team_alias": null, "team_id": "text", "organization_id": null, "admins": [],'
    r' "members": [], "members_with_roles": [], "team_member_permissions": [],'
    r' "metadata": {"guardrails": ["pii"]}, "tpm_limit": null, "rpm_limit": null,'
    r' "tpd_limit": null, "max_budget": null, "soft_budget": null, "budget_duration": null,'
    r' "budget_limits": [{"budget_duration": "1d", "max_budget": 1.0, "reset_at": null}],'
    r' "models": [], "blocked": false, "router_settings": {"num_retries": 2},'
    r' "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,'
    r' "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,'
    r' "model_spend": {}, "model_max_budget": {"claude": {"budget_limit": 2.0}},'
    r' "policies": [], "allow_team_guardrail_config": false, "litellm_model_table": null,'
    r' "object_permission": null, "object_permission_id": null,'
    r' "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00Z",'
    r' "last_refreshed_at": null}'
)


def _unstamped(entry: str) -> str:
    """``entry`` with the time it was cached taken out, as the Rust store leaves it for the host that caches the
    team to stamp."""
    return re.sub(r'"last_refreshed_at": [0-9.e+]+', '"last_refreshed_at": null', entry)


_TEAMS: Final = pytest.mark.parametrize(
    ("row", "entry"),
    [
        (CREATED_ROW, CREATED_ENTRY),
        (FULL_ROW, FULL_ENTRY),
        (NULLS_ROW, NULLS_ENTRY),
        (TEXT_ROW, TEXT_ENTRY),
        (DEFAULTS_ROW, DEFAULTS_ENTRY),
    ],
    ids=["as_created", "every_column_set", "json_nulls", "json_text", "upserted_with_defaults"],
)


@_TEAMS
def test_the_team_entry_is_what_python_dumps_for_the_repository_row(row: str, entry: str) -> None:
    """The row with the object permission it points at nested, as ``get_team_object`` nests it."""
    stored: Final = json.loads(row)
    permission: Final = json.loads(PERMISSION_ROW) if stored["object_permission_id"] else None
    team: Final = LiteLLM_TeamTableCachedObj.model_validate({**stored, "object_permission": permission})

    assert json.dumps(CacheCodec.serialize(team, model_type=LiteLLM_TeamTableCachedObj)) == entry


@_TEAMS
@pytest.mark.asyncio
async def test_the_proxy_caches_a_team_it_read_from_the_database_in_the_bytes_the_rust_store_caches(
    row: str, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_team_object`` reads the row, tries to nest the object permission it points at and caches the team with
    ``cache_team_object``, as the Rust store nests it."""
    stores: Final = install_rust_auth_stores(monkeypatch)
    stored: Final = json.loads(row)
    stores.team.rows[stored["team_id"]] = stored
    stores.object_permission.seed("permission", LiteLLM_ObjectPermissionTable.model_validate_json(PERMISSION_ROW))
    cache: Final = UserApiKeyCache()

    await get_team_object(team_id=stored["team_id"], prisma_client=MagicMock(), user_api_key_cache=cache)

    assert _unstamped(json.dumps(cache.get_local(AuthObject.team(stored["team_id"])))) == entry


@pytest.mark.parametrize(
    "entry",
    [CREATED_ENTRY, FULL_ENTRY, NULLS_ENTRY, TEXT_ENTRY, STAMPED_ENTRY, DEFAULTS_ENTRY],
    ids=[
        "as_created",
        "every_column_set",
        "json_nulls",
        "json_text",
        "stamped_by_the_host_that_cached_it",
        "upserted_with_defaults",
    ],
)
@pytest.mark.asyncio
async def test_python_caches_the_team_it_read_from_the_entry_in_the_same_bytes(entry: str) -> None:
    """A field added to ``LiteLLM_TeamTableCachedObj`` shows up here first: Python then writes a field the Rust record
    does not name, and the Rust side of this contract has to name it too."""
    cache: Final = UserApiKeyCache()
    team_id: Final = json.loads(entry)["team_id"]
    cache.set_local(AuthObject.team(team_id), json.loads(entry), None)

    team: Final = await get_team_object(
        team_id=team_id, prisma_client=MagicMock(), user_api_key_cache=cache, check_cache_only=True
    )
    await cache_team_object(team_id, team, cache, None)

    assert _unstamped(json.dumps(cache.get_local(AuthObject.team(team_id)))) == _unstamped(entry)
