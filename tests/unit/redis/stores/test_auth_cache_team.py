"""The auth cache entry the Rust auth store shapes for a team, against Python's own model and its own team reader and
writer. The row and entry literals are the ones ``litellm/rust/crates/auth-store/tests/team.rs`` reads from Postgres
and ``litellm/rust/crates/redis-store/tests/auth_cache_team.rs`` round-trips through Redis. Change all three
together."""

import json
import re
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.proxy._types import LiteLLM_TeamTableCachedObj
from litellm.proxy.auth.auth_checks import cache_team_object, get_team_object
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

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
    r'"default_team_member_models":["claude"],"budget_limits":[{"max_budget": 1,'
    r' "window_start": "2026-11-01T00:00:00+00:00", "budget_duration": "1d"},'
    r' {"max_budget": 2.5, "budget_duration": "30d"}],"model_id":7,'
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
    r' "budget_limits": [{"budget_duration": "1d", "max_budget": 1.0, "reset_at": null,'
    r' "window_start": "2026-11-01T00:00:00Z"}, {"budget_duration": "30d",'
    r' "max_budget": 2.5, "reset_at": null}], "models": ["claude", "all-proxy-models"],'
    r' "blocked": true, "router_settings": {"num_retries": 2},'
    r' "access_group_ids": ["group"], "default_team_member_models": ["claude"],'
    r' "spend": 1.5, "max_parallel_requests": 3, "budget_reset_at": "2026-12-01T00:00:00Z",'
    r' "model_id": 7, "model_spend": {"claude": 1.5},'
    r' "model_max_budget": {"claude": {"budget_limit": 2.0}}, "policies": ["policy"],'
    r' "allow_team_guardrail_config": true, "litellm_model_table": {"id": 7,'
    r' "model_aliases": {"gpt": "claude"}, "created_by": "test", "updated_by": "test",'
    r' "team": null}, "object_permission": null, "object_permission_id": "permission",'
    r' "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z",'
    r' "last_refreshed_at": null}'
)


def _unstamped(entry: str) -> str:
    """``entry`` with the time it was cached taken out, as the Rust store leaves it for the host that caches the
    team to stamp."""
    return re.sub(r'"last_refreshed_at": [0-9.e+]+', '"last_refreshed_at": null', entry)


_TEAMS: Final = pytest.mark.parametrize(
    ("row", "entry"),
    [(CREATED_ROW, CREATED_ENTRY), (FULL_ROW, FULL_ENTRY)],
    ids=["as_created", "every_column_set"],
)


@_TEAMS
def test_the_team_entry_is_what_python_dumps_for_the_repository_row(row: str, entry: str) -> None:
    team: Final = LiteLLM_TeamTableCachedObj.model_validate_json(row)

    assert json.dumps(CacheCodec.serialize(team, model_type=LiteLLM_TeamTableCachedObj)) == entry


@_TEAMS
@pytest.mark.asyncio
async def test_the_proxy_caches_a_team_it_read_from_the_database_in_the_bytes_the_rust_store_caches(
    row: str, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_team_object`` reads the row, tries to nest the object permission it points at and caches the team with
    ``cache_team_object``. No permission row exists here, so it nests nothing, as the Rust store never does."""
    stores: Final = install_rust_auth_stores(monkeypatch)
    stored: Final = json.loads(row)
    stores.team.rows[stored["team_id"]] = stored
    cache: Final = UserApiKeyCache()

    await get_team_object(team_id=stored["team_id"], prisma_client=MagicMock(), user_api_key_cache=cache)

    assert _unstamped(json.dumps(cache.get_local(AuthObject.team(stored["team_id"])))) == entry


@pytest.mark.parametrize("entry", [CREATED_ENTRY, FULL_ENTRY], ids=["as_created", "every_column_set"])
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

    assert _unstamped(json.dumps(cache.get_local(AuthObject.team(team_id)))) == entry
