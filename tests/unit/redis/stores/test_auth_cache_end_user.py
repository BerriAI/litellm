"""The auth cache entry the Rust auth store shapes for an end user, against Python's own model and its own end user
reader and writer. The row and entry literals are the ones ``litellm/rust/crates/auth-store/tests/end_user.rs`` reads
from Postgres and ``litellm/rust/crates/redis-store/tests/auth_cache_end_user.rs`` round-trips through Redis. Change
all three together."""

import json
from typing import Final
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.proxy._types import LiteLLM_EndUserTable
from litellm.proxy.auth.auth_checks import get_end_user_object
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from litellm.repositories import rust_database
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

CREATED_ROW: Final = (
    r'{"user_id":"created","alias":null,"spend":0.0,"allowed_model_region":null,"default_model":null,"models":[],'
    r'"budget_id":null,"litellm_budget_table":null,"object_permission_id":null,"object_permission":null,'
    r'"blocked":false}'
)
CREATED_ENTRY: Final = (
    r'{"user_id": "created", "blocked": false, "alias": null, "spend": 0.0, "allowed_model_region": null,'
    r' "default_model": null, "models": [], "budget_id": null, "litellm_budget_table": null,'
    r' "object_permission_id": null, "object_permission": null}'
)
FULL_ROW: Final = (
    r'{"user_id":"full","alias":"Full","spend":1.5,"allowed_model_region":"eu","default_model":"claude",'
    r'"models":["claude","gpt"],"budget_id":"budget","litellm_budget_table":{"budget_id":"budget","name":"Budget",'
    r'"max_budget":10.5,"rollover_max_budget":20.0,"soft_budget":8.0,"max_parallel_requests":3,"tpm_limit":1000,'
    r'"rpm_limit":10,"tpd_limit":100000,"model_max_budget":{"claude": {"max_budget": 1}},'
    r'"model_tpm_limit":{"gpt": "7", "claude": 100.0},"model_rpm_limit":{"claude": "5"},"budget_duration":"30d",'
    r'"budget_limits":[],"cap_policy":"hard","budget_reset_at":"2026-12-01T00:00:00+00:00","budget_reset_count":2,'
    r'"allowed_models":["claude"],"temp_budget_increase":1.5,"temp_budget_expiry":"2026-11-20T12:30:00.250+00:00",'
    r'"created_at":"2026-11-01T00:00:00+00:00","created_by":"test","updated_at":"2026-11-02T00:00:00+00:00",'
    r'"updated_by":"test"},"object_permission_id":"permission",'
    r'"object_permission":{"object_permission_id":"permission","mcp_servers":["server"],'
    r'"mcp_access_groups":["group"],"mcp_tool_permissions":{"server": ["tool"]},"vector_stores":["store"],'
    r'"agents":["agent"],"agent_access_groups":["agents"],"models":["claude"],"blocked_tools":["shell"],'
    r'"mcp_toolsets":["toolset"],"search_tools":["search"],"mcp_tool_search_enabled":true,"skills":["skill"]},'
    r'"blocked":true}'
)
FULL_ENTRY: Final = (
    r'{"user_id": "full", "blocked": true, "alias": "Full", "spend": 1.5, "allowed_model_region": "eu",'
    r' "default_model": "claude", "models": ["claude", "gpt"], "budget_id": "budget",'
    r' "litellm_budget_table": {"budget_id": "budget", "soft_budget": 8.0, "max_budget": 10.5,'
    r' "rollover_max_budget": 20.0, "max_parallel_requests": 3, "tpm_limit": 1000, "rpm_limit": 10,'
    r' "tpd_limit": 100000, "model_max_budget": {"claude": {"max_budget": 1}}, "model_tpm_limit": {"gpt": 7,'
    r' "claude": 100}, "model_rpm_limit": {"claude": 5}, "budget_duration": "30d", "allowed_models": ["claude"],'
    r' "temp_budget_increase": 1.5, "temp_budget_expiry": "2026-11-20T12:30:00.250000Z"},'
    r' "object_permission_id": "permission", "object_permission": {"object_permission_id": "permission",'
    r' "mcp_servers": ["server"], "mcp_access_groups": ["group"], "mcp_tool_permissions": {"server": ["tool"]},'
    r' "vector_stores": ["store"], "agents": ["agent"], "agent_access_groups": ["agents"], "models": ["claude"],'
    r' "mcp_toolsets": ["toolset"], "blocked_tools": ["shell"], "search_tools": ["search"],'
    r' "mcp_tool_search_enabled": true, "skills": ["skill"], "apps": null}}'
)
BARE_PERMISSION_ROW: Final = (
    r'{"user_id":"bare_permission","alias":null,"spend":0.0,"allowed_model_region":null,'
    r'"default_model":null,"models":[],"budget_id":null,"litellm_budget_table":null,'
    r'"object_permission_id":"bare","object_permission":{"object_permission_id":"bare",'
    r'"mcp_servers":[],"mcp_access_groups":[],"mcp_tool_permissions":null,"vector_stores":[],'
    r'"agents":[],"agent_access_groups":[],"models":[],"blocked_tools":[],"mcp_toolsets":[],'
    r'"search_tools":[],"mcp_tool_search_enabled":null,"skills":[]},"blocked":false}'
)
BARE_PERMISSION_ENTRY: Final = (
    r'{"user_id": "bare_permission", "blocked": false, "alias": null, "spend": 0.0,'
    r' "allowed_model_region": null, "default_model": null, "models": [], "budget_id": null,'
    r' "litellm_budget_table": null, "object_permission_id": "bare",'
    r' "object_permission": {"object_permission_id": "bare", "mcp_servers": [],'
    r' "mcp_access_groups": [], "mcp_tool_permissions": null, "vector_stores": [],'
    r' "agents": [], "agent_access_groups": [], "models": [], "mcp_toolsets": [],'
    r' "blocked_tools": [], "search_tools": [], "mcp_tool_search_enabled": null, "skills": [],'
    r' "apps": null}}'
)
MERGED_ENTRY: Final = (
    r'{"user_id": "created", "blocked": false, "alias": null, "spend": 0.0, "allowed_model_region": null,'
    r' "default_model": null, "models": [], "budget_id": null, "litellm_budget_table": {"budget_id": "default",'
    r' "soft_budget": null, "max_budget": 5.0, "max_parallel_requests": null, "tpm_limit": null, "rpm_limit": null,'
    r' "tpd_limit": null, "model_max_budget": null, "model_tpm_limit": null, "model_rpm_limit": null,'
    r' "budget_duration": null, "allowed_models": null, "temp_budget_increase": null, "temp_budget_expiry": null},'
    r' "object_permission_id": null, "object_permission": null}'
)
DEFAULT_BUDGET_ROW: Final = r'{"budget_id":"default","max_budget":5.0}'

_FIELDS: Final = TypeAdapter(dict[str, object])
_END_USERS: Final = pytest.mark.parametrize(
    ("row", "entry"),
    [(CREATED_ROW, CREATED_ENTRY), (FULL_ROW, FULL_ENTRY), (BARE_PERMISSION_ROW, BARE_PERMISSION_ENTRY)],
    ids=["as_created", "every_column_set", "a_permission_with_null_arrays"],
)

_FULL: Final = json.loads(FULL_ROW)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowed_model_region", "ap"),
        (
            "litellm_budget_table",
            {**_FULL["litellm_budget_table"], "budget_limits": [{"budget_duration": "1d", "max_budget": 1}]},
        ),
        ("object_permission", {**_FULL["object_permission"], "mcp_tool_permissions": {"server": "tool"}}),
    ],
    ids=["a_region_the_model_does_not_take", "a_budget_with_concurrent_windows", "a_permission_the_model_refuses"],
)
def test_python_refuses_each_row_the_rust_store_refuses(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        LiteLLM_EndUserTable.model_validate({**_FULL, field: value})


@_END_USERS
def test_the_end_user_entry_is_what_python_dumps_for_the_repository_row(row: str, entry: str) -> None:
    end_user: Final = rust_database.decode(LiteLLM_EndUserTable, row)
    assert end_user is not None

    assert json.dumps(CacheCodec.serialize(end_user, model_type=LiteLLM_EndUserTable)) == entry


@_END_USERS
@pytest.mark.asyncio
async def test_the_proxy_caches_an_end_user_it_read_from_the_database_in_the_bytes_the_rust_store_caches(
    row: str, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token budget for the end user makes ``get_end_user_object`` read even a row that restricts nothing."""
    stores: Final = install_rust_auth_stores(monkeypatch)
    end_user_id: Final = json.loads(row)["user_id"]
    stores.end_user.rows[end_user_id] = _FIELDS.validate_json(row)
    cache: Final = UserApiKeyCache()

    await get_end_user_object(
        end_user_id=end_user_id, prisma_client=MagicMock(), user_api_key_cache=cache, token_end_user_max_budget=1.0
    )

    assert json.dumps(cache.get_local(AuthObject.end_user(end_user_id))) == entry


@pytest.mark.asyncio
async def test_the_proxy_caches_an_end_user_without_a_budget_with_the_default_end_user_budget_merged_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The merge happens before the cache write, so the entry holds a budget the row does not name. The Rust store
    returns the row as it is, and the host that caches it owns the merge."""
    stores: Final = install_rust_auth_stores(monkeypatch)
    stores.end_user.rows["created"] = _FIELDS.validate_json(CREATED_ROW)
    stores.budget.rows["default"] = _FIELDS.validate_json(DEFAULT_BUDGET_ROW)
    monkeypatch.setattr(litellm, "max_end_user_budget_id", "default")
    cache: Final = UserApiKeyCache()

    await get_end_user_object(end_user_id="created", prisma_client=MagicMock(), user_api_key_cache=cache)

    assert json.dumps(cache.get_local(AuthObject.end_user("created"))) == MERGED_ENTRY


@pytest.mark.parametrize(
    "entry",
    [CREATED_ENTRY, FULL_ENTRY, BARE_PERMISSION_ENTRY, MERGED_ENTRY],
    ids=["as_created", "every_column_set", "a_permission_with_null_arrays", "the_default_budget_merged_in"],
)
@pytest.mark.asyncio
async def test_python_caches_the_end_user_it_read_from_the_entry_in_the_same_bytes(entry: str) -> None:
    """A field added to ``LiteLLM_EndUserTable`` or the budget or object permission it nests shows up here first:
    Python then writes a field the Rust record does not name, and the Rust side of this contract has to name it
    too."""
    cache: Final = UserApiKeyCache()
    end_user_id: Final = json.loads(entry)["user_id"]
    cache.set_local(AuthObject.end_user(end_user_id), _FIELDS.validate_json(entry), None)

    end_user: Final = await get_end_user_object(
        end_user_id=end_user_id, prisma_client=MagicMock(), user_api_key_cache=cache
    )

    assert json.dumps(CacheCodec.serialize(end_user, model_type=LiteLLM_EndUserTable)) == entry
