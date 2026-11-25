"""The auth cache entries the Rust auth store writes for a virtual key and its object permission, against Python's
own models and its own key writer. The row and entry literals are the ones
``litellm/rust/crates/auth-store/tests/key.rs`` reads from Postgres and finds in Redis, so the store and Python
are pinned to the same bytes. Change both together."""

import json
import re
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._types import LiteLLM_VerificationTokenView, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import cache_key_object, get_key_object
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

STAMP: Final = 1763546400.5

BARE_ROW: Final = (
    r'{"token":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","key_name":"sk-...abcd",'
    r'"key_alias":null,"soft_budget_cooldown":false,"spend":0.0,"total_spend":0.0,"expires":null,'
    r'"models":[],"aliases":{},"config":{},"router_settings":{},"user_id":null,"team_id":null,'
    r'"agent_id":null,"project_id":null,"permissions":{},"max_parallel_requests":null,"metadata":{},'
    r'"blocked":null,"tpm_limit":null,"rpm_limit":null,"tpd_limit":null,"max_budget":null,'
    r'"soft_budget":null,"budget_duration":null,"budget_reset_at":null,"allowed_cache_controls":[],'
    r'"allowed_routes":[],"key_type":null,"access_group_ids":[],"model_spend":{},"model_max_budget":{},'
    r'"budget_fallbacks":{},"budget_id":null,"org_id":null,"object_permission_id":null,'
    r'"created_at":"2026-11-19T10:00:00.123+00:00","created_by":"default_user_id",'
    r'"updated_at":"2026-11-19T10:00:00+00:00","updated_by":"default_user_id","settings_updated_at":null,'
    r'"last_active":null,"rotation_count":0,"auto_rotate":false,"rotation_interval":null,'
    r'"last_rotation_at":null,"key_rotation_at":null,"budget_limits":null,"team_spend":null,'
    r'"team_alias":null,"team_tpm_limit":null,"team_rpm_limit":null,"team_tpd_limit":null,'
    r'"team_max_budget":null,"team_soft_budget":null,"team_model_max_budget":null,"team_models":[],'
    r'"team_blocked":false,"team_model_aliases":null,"team_member":null,"team_metadata":null,'
    r'"team_object_permission_id":null,"team_member_spend":null,"team_member_tpm_limit":null,'
    r'"team_member_rpm_limit":null,"organization_alias":null,"organization_max_budget":null,'
    r'"organization_tpm_limit":null,"organization_rpm_limit":null,"organization_metadata":null,'
    r'"project_alias":null,"last_refreshed_at":1763546400.5}'
)
BARE_ENTRY: Final = (
    r'{"key_id": null, "token": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    r' "key_name": "sk-...abcd", "key_alias": null, "spend": 0.0, "total_spend": 0.0, "max_budget": null,'
    r' "expires": null, "models": [], "aliases": {}, "config": {}, "user_id": null, "team_id": null,'
    r' "agent_id": null, "project_id": null, "max_parallel_requests": null, "metadata": {},'
    r' "tpm_limit": null, "rpm_limit": null, "tpd_limit": null, "budget_duration": null,'
    r' "budget_reset_at": null, "allowed_cache_controls": [], "allowed_routes": [], "key_type": null,'
    r' "permissions": {}, "model_spend": {}, "model_max_budget": {}, "budget_fallbacks": {},'
    r' "soft_budget_cooldown": false, "blocked": null, "litellm_budget_table": null, "budget_id": null,'
    r' "org_id": null, "created_at": "2026-11-19T10:00:00.123000Z", "created_by": "default_user_id",'
    r' "updated_at": "2026-11-19T10:00:00Z", "updated_by": "default_user_id", "settings_updated_at": null,'
    r' "last_active": null, "object_permission_id": null, "object_permission": null,'
    r' "access_group_ids": [], "rotation_count": 0, "auto_rotate": false, "rotation_interval": null,'
    r' "last_rotation_at": null, "key_rotation_at": null, "router_settings": {}, "budget_limits": null,'
    r' "team_spend": null, "team_alias": null, "team_tpm_limit": null, "team_rpm_limit": null,'
    r' "team_tpd_limit": null, "team_max_budget": null, "team_soft_budget": null,'
    r' "team_model_max_budget": null, "team_models": [], "team_blocked": false, "soft_budget": null,'
    r' "team_model_aliases": null, "team_member": null, "team_metadata": null,'
    r' "team_object_permission_id": null, "team_member_spend": null, "team_member_tpm_limit": null,'
    r' "team_member_rpm_limit": null, "end_user_id": null, "end_user_tpm_limit": null,'
    r' "end_user_rpm_limit": null, "end_user_tpd_limit": null, "end_user_max_budget": null,'
    r' "end_user_model_max_budget": null, "end_user_model_tpm_limit": null,'
    r' "end_user_model_rpm_limit": null, "customer_hierarchy": null, "organization_alias": null,'
    r' "organization_max_budget": null, "organization_tpm_limit": null, "organization_rpm_limit": null,'
    r' "organization_metadata": null, "project_alias": null, "project_metadata": null,'
    r' "last_refreshed_at": 1763546400.5, "api_key": null, "user_role": null,'
    r' "allowed_model_region": null, "parent_otel_span": null, "rpm_limit_per_model": null,'
    r' "tpm_limit_per_model": null, "user_tpm_limit": null, "user_rpm_limit": null, "user_email": null,'
    r' "user_spend": null, "user_max_budget": null, "user_model_max_budget": null, "request_route": null,'
    r' "is_session_token": false, "user": null, "created_by_user": null,'
    r' "end_user_object_permission": null, "team_object_permission": null, "jwt_claims": null}'
)
TEAM_ROW: Final = (
    r'{"token":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","key_name":"sk-...wxyz",'
    r'"key_alias":"team key","soft_budget_cooldown":false,"spend":0.30000000000000004,"total_spend":0.0,'
    r'"expires":"2030-01-02T03:04:05.678+00:00","models":["m1","gpt-4o"],"aliases":{},"config":{},'
    r'"router_settings":null,"user_id":"kv-user","team_id":"kv-team","agent_id":null,"project_id":null,'
    r'"permissions":{},"max_parallel_requests":3,"metadata":{"a": [1, 2.5], "note": "é"},"blocked":null,'
    r'"tpm_limit":77,"rpm_limit":5,"tpd_limit":null,"max_budget":9.5,"soft_budget":4.5,'
    r'"budget_duration":null,"budget_reset_at":"2030-01-02T00:00:00+00:00","allowed_cache_controls":null,'
    r'"allowed_routes":null,"key_type":null,"access_group_ids":[],"model_spend":{},'
    r'"model_max_budget":{"gpt-4o": 1},"budget_fallbacks":{},"budget_id":"kv-budget","org_id":null,'
    r'"object_permission_id":"kv-permission","created_at":"2026-11-19T10:00:00.123+00:00",'
    r'"created_by":null,"updated_at":"2026-11-19T10:00:00+00:00","updated_by":null,'
    r'"settings_updated_at":null,"last_active":null,"rotation_count":null,"auto_rotate":null,'
    r'"rotation_interval":null,"last_rotation_at":null,"key_rotation_at":null,"budget_limits":null,'
    r'"team_spend":2.25,"team_alias":"Team KV","team_tpm_limit":null,"team_rpm_limit":null,'
    r'"team_tpd_limit":null,"team_max_budget":12.0,"team_soft_budget":null,"team_model_max_budget":{},'
    r'"team_models":["gpt-4o"],"team_blocked":true,"team_model_aliases":null,'
    r'"team_member":{"role": "admin", "joined": "2026", "user_id": "kv-user"},'
    r'"team_metadata":{"tier": "gold"},"team_object_permission_id":null,"team_member_spend":null,'
    r'"team_member_tpm_limit":null,"team_member_rpm_limit":null,"organization_alias":null,'
    r'"organization_max_budget":null,"organization_tpm_limit":null,"organization_rpm_limit":null,'
    r'"organization_metadata":null,"project_alias":null,"last_refreshed_at":1763546400.5}'
)
TEAM_ENTRY: Final = (
    r'{"key_id": null, "token": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
    r' "key_name": "sk-...wxyz", "key_alias": "team key", "spend": 0.30000000000000004,'
    r' "total_spend": 0.0, "max_budget": 9.5, "expires": "2030-01-02T03:04:05.678+00:00", "models": ["m1",'
    r' "gpt-4o"], "aliases": {}, "config": {}, "user_id": "kv-user", "team_id": "kv-team",'
    r' "agent_id": null, "project_id": null, "max_parallel_requests": 3, "metadata": {"a": [1, 2.5],'
    r' "note": "\u00e9"}, "tpm_limit": 77, "rpm_limit": 5, "tpd_limit": null, "budget_duration": null,'
    r' "budget_reset_at": "2030-01-02T00:00:00Z", "allowed_cache_controls": [], "allowed_routes": [],'
    r' "key_type": null, "permissions": {}, "model_spend": {}, "model_max_budget": {"gpt-4o": 1},'
    r' "budget_fallbacks": {}, "soft_budget_cooldown": false, "blocked": null,'
    r' "litellm_budget_table": null, "budget_id": "kv-budget", "org_id": null,'
    r' "created_at": "2026-11-19T10:00:00.123000Z", "created_by": null,'
    r' "updated_at": "2026-11-19T10:00:00Z", "updated_by": null, "settings_updated_at": null,'
    r' "last_active": null, "object_permission_id": "kv-permission",'
    r' "object_permission": {"object_permission_id": "kv-permission", "mcp_servers": ["server-a"],'
    r' "mcp_access_groups": [], "mcp_tool_permissions": {"server-a": ["search"]}, "mcp_event_permissions": null,'
    r' "vector_stores": ["store-a"], "agents": [], "agent_access_groups": [], "models": [],'
    r' "mcp_toolsets": [], "blocked_tools": [], "search_tools": [], "mcp_tool_search_enabled": null,'
    r' "skills": [], "apps": null}, "access_group_ids": [], "rotation_count": 0, "auto_rotate": false,'
    r' "rotation_interval": null, "last_rotation_at": null, "key_rotation_at": null,'
    r' "router_settings": null, "budget_limits": null, "team_spend": 2.25, "team_alias": "Team KV",'
    r' "team_tpm_limit": null, "team_rpm_limit": null, "team_tpd_limit": null, "team_max_budget": 12.0,'
    r' "team_soft_budget": null, "team_model_max_budget": {}, "team_models": ["gpt-4o"],'
    r' "team_blocked": true, "soft_budget": 4.5, "team_model_aliases": null,'
    r' "team_member": {"user_id": "kv-user", "user_email": null, "role": "admin"},'
    r' "team_metadata": {"tier": "gold"}, "team_object_permission_id": null, "team_member_spend": null,'
    r' "team_member_tpm_limit": null, "team_member_rpm_limit": null, "end_user_id": null,'
    r' "end_user_tpm_limit": null, "end_user_rpm_limit": null, "end_user_tpd_limit": null,'
    r' "end_user_max_budget": null, "end_user_model_max_budget": null, "end_user_model_tpm_limit": null,'
    r' "end_user_model_rpm_limit": null, "customer_hierarchy": null, "organization_alias": null,'
    r' "organization_max_budget": null, "organization_tpm_limit": null, "organization_rpm_limit": null,'
    r' "organization_metadata": null, "project_alias": null, "project_metadata": null,'
    r' "last_refreshed_at": 1763546400.5, "api_key": null, "user_role": null,'
    r' "allowed_model_region": null, "parent_otel_span": null, "rpm_limit_per_model": null,'
    r' "tpm_limit_per_model": null, "user_tpm_limit": null, "user_rpm_limit": null, "user_email": null,'
    r' "user_spend": null, "user_max_budget": null, "user_model_max_budget": null, "request_route": null,'
    r' "is_session_token": false, "user": null, "created_by_user": null,'
    r' "end_user_object_permission": null, "team_object_permission": null, "jwt_claims": null}'
)
PERMISSION_ROW: Final = (
    r'{"object_permission_id":"kv-permission","mcp_servers":["server-a"],"mcp_access_groups":[],'
    r'"mcp_tool_permissions":{"server-a": ["search"]},"vector_stores":["store-a"],"agents":[],'
    r'"agent_access_groups":[],"models":[],"blocked_tools":[],"mcp_toolsets":[],"search_tools":[],'
    r'"mcp_tool_search_enabled":null,"skills":[]}'
)
PERMISSION_ENTRY: Final = (
    r'{"object_permission_id": "kv-permission", "mcp_servers": ["server-a"], "mcp_access_groups": [],'
    r' "mcp_tool_permissions": {"server-a": ["search"]}, "mcp_event_permissions": null,'
    r' "vector_stores": ["store-a"], "agents": [],'
    r' "agent_access_groups": [], "models": [], "mcp_toolsets": [], "blocked_tools": [],'
    r' "search_tools": [], "mcp_tool_search_enabled": null, "skills": [], "apps": null}'
)
FULL_ROW: Final = (
    r'{"token":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","key_name":"sk-...full",'
    r'"key_alias":"full key","soft_budget_cooldown":true,"spend":1.25,"total_spend":2.5,'
    r'"expires":"2031-02-03T04:05:06+00:00","models":["k-model"],"aliases":{"alias": "k-model"},'
    r'"config":{"cfg": 1},"router_settings":{"routing_strategy": "simple-shuffle"},"user_id":"full-user",'
    r'"team_id":"full-team","agent_id":"full-agent","project_id":"full-project",'
    r'"permissions":{"perm": true},"max_parallel_requests":7,"metadata":{"meta": [1]},"blocked":false,'
    r'"tpm_limit":101,"rpm_limit":102,"tpd_limit":103,"max_budget":104.5,"soft_budget":5.5,'
    r'"budget_duration":"30d","budget_reset_at":"2031-03-01T00:00:00.500+00:00",'
    r'"allowed_cache_controls":["no-cache"],"allowed_routes":["/v1/messages"],"key_type":"llm_api",'
    r'"access_group_ids":["group-a"],"model_spend":{"k-model": 0.5},'
    r'"model_max_budget":{"k-model": {"budget_limit": 2}},"budget_fallbacks":{"k-model": ["fallback"]},'
    r'"budget_id":"full-key-budget","org_id":"full-org","object_permission_id":"full-key-permission",'
    r'"created_at":"2026-01-01T01:01:01.001+00:00","created_by":"creator",'
    r'"updated_at":"2026-02-02T02:02:02.002+00:00","updated_by":"updater",'
    r'"settings_updated_at":"2026-03-03T03:03:03+00:00","last_active":"2026-04-04T04:04:04.400+00:00",'
    r'"rotation_count":3,"auto_rotate":true,"rotation_interval":"90d",'
    r'"last_rotation_at":"2026-05-05T05:05:05+00:00","key_rotation_at":"2026-08-08T08:08:08+00:00",'
    r'"budget_limits":[{"model": "k-model", "budget": 1}],"team_spend":3.5,"team_alias":"Full Team",'
    r'"team_tpm_limit":61,"team_rpm_limit":62,"team_tpd_limit":63,"team_max_budget":51.5,'
    r'"team_soft_budget":52.5,"team_model_max_budget":{"t-model": {"budget_limit": 3}},'
    r'"team_models":["t-model"],"team_blocked":false,"team_model_aliases":{"fast": "t-model"},'
    r'"team_member":{"role": "user", "user_id": "full-user", "user_email": "user@example.com"},'
    r'"team_metadata":{"team": true},"team_object_permission_id":"full-team-permission",'
    r'"team_member_spend":4.5,"team_member_tpm_limit":21,"team_member_rpm_limit":22,'
    r'"organization_alias":"Full Org","organization_max_budget":43.5,"organization_tpm_limit":31,'
    r'"organization_rpm_limit":32,"organization_metadata":{"org": true},"project_alias":"Full Project",'
    r'"last_refreshed_at":1763546400.5}'
)
FULL_ENTRY: Final = (
    r'{"key_id": null, "token": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
    r' "key_name": "sk-...full", "key_alias": "full key", "spend": 1.25, "total_spend": 2.5,'
    r' "max_budget": 104.5, "expires": "2031-02-03T04:05:06+00:00", "models": ["k-model"],'
    r' "aliases": {"alias": "k-model"}, "config": {"cfg": 1}, "user_id": "full-user",'
    r' "team_id": "full-team", "agent_id": "full-agent", "project_id": "full-project",'
    r' "max_parallel_requests": 7, "metadata": {"meta": [1]}, "tpm_limit": 101, "rpm_limit": 102,'
    r' "tpd_limit": 103, "budget_duration": "30d", "budget_reset_at": "2031-03-01T00:00:00.500000Z",'
    r' "allowed_cache_controls": ["no-cache"], "allowed_routes": ["/v1/messages"], "key_type": "llm_api",'
    r' "permissions": {"perm": true}, "model_spend": {"k-model": 0.5},'
    r' "model_max_budget": {"k-model": {"budget_limit": 2}},'
    r' "budget_fallbacks": {"k-model": ["fallback"]}, "soft_budget_cooldown": true, "blocked": false,'
    r' "litellm_budget_table": null, "budget_id": "full-key-budget", "org_id": "full-org",'
    r' "created_at": "2026-01-01T01:01:01.001000Z", "created_by": "creator",'
    r' "updated_at": "2026-02-02T02:02:02.002000Z", "updated_by": "updater",'
    r' "settings_updated_at": "2026-03-03T03:03:03Z", "last_active": "2026-04-04T04:04:04.400000Z",'
    r' "object_permission_id": "full-key-permission",'
    r' "object_permission": {"object_permission_id": "full-key-permission", "mcp_servers": ["server"],'
    r' "mcp_access_groups": ["mcp-group"], "mcp_tool_permissions": {"server": ["tool"]},'
    r' "mcp_event_permissions": {"server": ["build.*"]},'
    r' "vector_stores": ["store"], "agents": ["agent"], "agent_access_groups": ["agent-group"],'
    r' "models": ["p-model"], "mcp_toolsets": ["toolset"], "blocked_tools": ["blocked-tool"],'
    r' "search_tools": ["search-tool"], "mcp_tool_search_enabled": true, "skills": ["skill"],'
    r' "apps": null}, "access_group_ids": ["group-a"], "rotation_count": 3, "auto_rotate": true,'
    r' "rotation_interval": "90d", "last_rotation_at": "2026-05-05T05:05:05Z",'
    r' "key_rotation_at": "2026-08-08T08:08:08Z",'
    r' "router_settings": {"routing_strategy": "simple-shuffle"}, "budget_limits": [{"model": "k-model",'
    r' "budget": 1}], "team_spend": 3.5, "team_alias": "Full Team", "team_tpm_limit": 61,'
    r' "team_rpm_limit": 62, "team_tpd_limit": 63, "team_max_budget": 51.5, "team_soft_budget": 52.5,'
    r' "team_model_max_budget": {"t-model": {"budget_limit": 3}}, "team_models": ["t-model"],'
    r' "team_blocked": false, "soft_budget": 5.5, "team_model_aliases": {"fast": "t-model"},'
    r' "team_member": {"user_id": "full-user", "user_email": "user@example.com", "role": "user"},'
    r' "team_metadata": {"team": true}, "team_object_permission_id": "full-team-permission",'
    r' "team_member_spend": 4.5, "team_member_tpm_limit": 21, "team_member_rpm_limit": 22,'
    r' "end_user_id": null, "end_user_tpm_limit": null, "end_user_rpm_limit": null,'
    r' "end_user_tpd_limit": null, "end_user_max_budget": null, "end_user_model_max_budget": null,'
    r' "end_user_model_tpm_limit": null, "end_user_model_rpm_limit": null, "customer_hierarchy": null,'
    r' "organization_alias": "Full Org", "organization_max_budget": 43.5, "organization_tpm_limit": 31,'
    r' "organization_rpm_limit": 32, "organization_metadata": {"org": true},'
    r' "project_alias": "Full Project", "project_metadata": null, "last_refreshed_at": 1763546400.5,'
    r' "api_key": null, "user_role": null, "allowed_model_region": null, "parent_otel_span": null,'
    r' "rpm_limit_per_model": null, "tpm_limit_per_model": null, "user_tpm_limit": null,'
    r' "user_rpm_limit": null, "user_email": null, "user_spend": null, "user_max_budget": null,'
    r' "user_model_max_budget": null, "request_route": null, "is_session_token": false, "user": null,'
    r' "created_by_user": null, "end_user_object_permission": null, "team_object_permission": null,'
    r' "jwt_claims": null}'
)
FULL_PERMISSION_ROW: Final = (
    r'{"object_permission_id":"full-key-permission","mcp_servers":["server"],'
    r'"mcp_access_groups":["mcp-group"],"mcp_tool_permissions":{"server": ["tool"]},'
    r'"mcp_event_permissions":{"server": ["build.*"]},'
    r'"vector_stores":["store"],"agents":["agent"],"agent_access_groups":["agent-group"],'
    r'"models":["p-model"],"blocked_tools":["blocked-tool"],"mcp_toolsets":["toolset"],'
    r'"search_tools":["search-tool"],"mcp_tool_search_enabled":true,"skills":["skill"]}'
)
FULL_PERMISSION_ENTRY: Final = (
    r'{"object_permission_id": "full-key-permission", "mcp_servers": ["server"],'
    r' "mcp_access_groups": ["mcp-group"], "mcp_tool_permissions": {"server": ["tool"]},'
    r' "mcp_event_permissions": {"server": ["build.*"]},'
    r' "vector_stores": ["store"], "agents": ["agent"], "agent_access_groups": ["agent-group"],'
    r' "models": ["p-model"], "mcp_toolsets": ["toolset"], "blocked_tools": ["blocked-tool"],'
    r' "search_tools": ["search-tool"], "mcp_tool_search_enabled": true, "skills": ["skill"],'
    r' "apps": null}'
)


def _stamped(entry: str) -> str:
    """``entry`` with the time it was cached replaced by ``STAMP``, the one field no two runs agree on."""
    return re.sub(r'"last_refreshed_at": [0-9.e+]+', f'"last_refreshed_at": {STAMP}', entry)


_KEYS: Final = pytest.mark.parametrize(
    ("row", "entry", "permission_row"),
    [(BARE_ROW, BARE_ENTRY, None), (TEAM_ROW, TEAM_ENTRY, PERMISSION_ROW), (FULL_ROW, FULL_ENTRY, FULL_PERMISSION_ROW)],
    ids=["as_generated", "in_a_team", "every_column_set"],
)


@_KEYS
def test_the_key_entry_is_what_main_cached_for_the_repository_row(
    row: str, entry: str, permission_row: str | None
) -> None:
    """``get_key_object`` on ``main`` validated the row through the view model, dropped its nulls, attached the
    object permission and stamped the time before ``cache_key_object`` dumped it."""
    key: Final = UserAPIKeyAuth.model_validate(
        LiteLLM_VerificationTokenView.model_validate_json(row).model_dump(exclude_none=True)
    )
    if permission_row is not None:
        key.object_permission = LiteLLM_ObjectPermissionTable.model_validate_json(permission_row)
    key.last_refreshed_at = STAMP

    assert json.dumps(CacheCodec.serialize(key, model_type=UserAPIKeyAuth)) == entry


@_KEYS
@pytest.mark.asyncio
async def test_the_proxy_caches_a_key_it_read_from_the_database_in_the_bytes_the_rust_store_caches(
    row: str, entry: str, permission_row: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit tests read keys through ``FakeKeyReader`` instead of the Rust repository, so it has to shape a row as
    the repository does, and the proxy nests the object permission itself. What it then caches has to be the entry
    the Rust store caches for the same rows, or the two hosts would disagree on a key they share."""
    stores: Final = install_rust_auth_stores(monkeypatch)
    stored: Final = json.loads(row)
    stores.key.rows[stored["token"]] = stored
    if permission_row is not None:
        permission: Final = LiteLLM_ObjectPermissionTable.model_validate_json(permission_row)
        stores.object_permission.seed(permission.object_permission_id, permission)
    cache: Final = UserApiKeyCache()

    await get_key_object(hashed_token=stored["token"], prisma_client=MagicMock(), user_api_key_cache=cache)

    assert _stamped(json.dumps(cache.get_local(AuthObject.virtual_key(stored["token"])))) == entry


@_KEYS
@pytest.mark.asyncio
async def test_python_caches_the_key_it_read_from_the_entry_in_the_same_bytes(
    row: str, entry: str, permission_row: str | None
) -> None:
    """A field added to ``UserAPIKeyAuth`` shows up here first: Python then writes a field the Rust record does not
    name, and the Rust side of this contract has to name it too."""
    cache: Final = UserApiKeyCache()
    key: Final = UserAPIKeyAuth.model_validate(json.loads(entry))
    assert key.token is not None

    await cache_key_object(key.token, key, cache, None)

    assert _stamped(json.dumps(cache.get_local(AuthObject.virtual_key(key.token)))) == entry


@pytest.mark.parametrize(
    ("row", "entry"),
    [(PERMISSION_ROW, PERMISSION_ENTRY), (FULL_PERMISSION_ROW, FULL_PERMISSION_ENTRY)],
    ids=["its_defaults", "every_column_set"],
)
def test_the_object_permission_entry_is_what_python_caches_for_the_repository_row(row: str, entry: str) -> None:
    permission: Final = LiteLLM_ObjectPermissionTable.model_validate_json(row)

    assert json.dumps(CacheCodec.serialize(permission, model_type=LiteLLM_ObjectPermissionTable)) == entry
