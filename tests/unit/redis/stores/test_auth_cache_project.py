"""The auth cache entry the Rust auth store shapes for a project, against Python's own model and its own project
reader and writer. The row and entry literals are the ones ``litellm/rust/crates/auth-store/tests/project.rs`` reads
from Postgres and ``litellm/rust/crates/redis-store/tests/auth_cache_project.rs`` round-trips through Redis. Change all
three together."""

import json
import re
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.proxy._types import LiteLLM_ProjectTable, LiteLLM_ProjectTableCachedObj
from litellm.proxy.auth.auth_checks import get_project_object
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from litellm.repositories import rust_database
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

CREATED_ROW: Final = (
    r'{"project_id":"created","project_alias":null,"description":null,"team_id":null,"budget_id":null,'
    r'"metadata":{},"models":[],"spend":0.0,"model_spend":{},"model_rpm_limit":{},"model_tpm_limit":{},'
    r'"blocked":false,"object_permission_id":null,"created_at":"2026-11-19T10:00:00.123+00:00",'
    r'"created_by":"creator","updated_at":"2026-11-19T10:00:00+00:00","updated_by":"creator",'
    r'"litellm_budget_table":null}'
)
CREATED_ENTRY: Final = (
    r'{"project_id": "created", "project_alias": null, "description": null, "team_id": null,'
    r' "budget_id": null, "metadata": {}, "models": [], "spend": 0.0, "model_spend": {},'
    r' "model_rpm_limit": {}, "model_tpm_limit": {}, "blocked": false, "object_permission_id": null,'
    r' "created_by": "creator", "updated_by": "creator", "created_at": "2026-11-19T10:00:00.123000Z",'
    r' "updated_at": "2026-11-19T10:00:00Z", "litellm_budget_table": null, "object_permission": null,'
    r' "last_refreshed_at": null}'
)
BARE_BUDGET_ROW: Final = (
    r'{"project_id":"bare_budget","project_alias":null,"description":null,"team_id":null,"budget_id":"bare",'
    r'"metadata":{},"models":[],"spend":0.0,"model_spend":{},"model_rpm_limit":{},"model_tpm_limit":{},'
    r'"blocked":false,"object_permission_id":null,"created_at":"2026-11-19T10:00:00+00:00","created_by":"creator",'
    r'"updated_at":"2026-11-19T10:00:00+00:00","updated_by":"creator","litellm_budget_table":{"budget_id":"bare",'
    r'"name":null,"max_budget":null,"soft_budget":null,"max_parallel_requests":null,"tpm_limit":null,'
    r'"rpm_limit":null,"tpd_limit":null,"model_max_budget":null,"model_tpm_limit":null,"model_rpm_limit":null,'
    r'"budget_duration":null,"budget_limits":null,"cap_policy":null,"budget_reset_at":null,"budget_reset_count":0,'
    r'"allowed_models":[],"temp_budget_increase":null,"temp_budget_expiry":null,'
    r'"created_at":"2026-11-01T00:00:00+00:00","created_by":"test","updated_at":"2026-11-01T00:00:00+00:00",'
    r'"updated_by":"test"}}'
)
BARE_BUDGET_ENTRY: Final = (
    r'{"project_id": "bare_budget", "project_alias": null, "description": null, "team_id": null,'
    r' "budget_id": "bare", "metadata": {}, "models": [], "spend": 0.0, "model_spend": {}, "model_rpm_limit": {},'
    r' "model_tpm_limit": {}, "blocked": false, "object_permission_id": null, "created_by": "creator",'
    r' "updated_by": "creator", "created_at": "2026-11-19T10:00:00Z", "updated_at": "2026-11-19T10:00:00Z",'
    r' "litellm_budget_table": {"budget_id": "bare", "soft_budget": null, "max_budget": null,'
    r' "max_parallel_requests": null, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,'
    r' "model_max_budget": null, "model_tpm_limit": null, "model_rpm_limit": null, "budget_duration": null,'
    r' "allowed_models": [], "temp_budget_increase": null, "temp_budget_expiry": null}, "object_permission": null,'
    r' "last_refreshed_at": null}'
)
FULL_ROW: Final = (
    r'{"project_id":"full","project_alias":"Full","description":"Every column set","team_id":"team",'
    r'"budget_id":"budget","metadata":{"guardrails": ["pii"]},"models":["claude","all-proxy-models"],'
    r'"spend":1.5,"model_spend":{"claude": 1.5},"model_rpm_limit":null,"model_tpm_limit":{"claude": 100},'
    r'"blocked":true,"object_permission_id":"permission","created_at":"2026-11-19T10:00:00.123+00:00",'
    r'"created_by":"creator","updated_at":"2026-11-19T10:00:00+00:00","updated_by":"updater",'
    r'"litellm_budget_table":{"budget_id":"budget","name":"Budget","max_budget":10.5,'
    r'"rollover_max_budget":20.0,"soft_budget":8.0,"max_parallel_requests":3,"tpm_limit":1000,"rpm_limit":10,'
    r'"tpd_limit":100000,"model_max_budget":{"claude": {"max_budget": 1}},"model_tpm_limit":{"gpt": "7",'
    r' "claude": 100.0},"model_rpm_limit":{"claude": "5"},"budget_duration":"30d","budget_limits":[],'
    r'"cap_policy":"hard","budget_reset_at":"2026-12-01T00:00:00+00:00","budget_reset_count":2,'
    r'"allowed_models":["claude"],"temp_budget_increase":1.5,'
    r'"temp_budget_expiry":"2026-11-20T12:30:00.250+00:00","created_at":"2026-11-01T00:00:00+00:00",'
    r'"created_by":"test","updated_at":"2026-11-02T00:00:00+00:00","updated_by":"test"}}'
)
FULL_ENTRY: Final = (
    r'{"project_id": "full", "project_alias": "Full", "description": "Every column set", "team_id": "team",'
    r' "budget_id": "budget", "metadata": {"guardrails": ["pii"]}, "models": ["claude", "all-proxy-models"],'
    r' "spend": 1.5, "model_spend": {"claude": 1.5}, "model_rpm_limit": null,'
    r' "model_tpm_limit": {"claude": 100}, "blocked": true, "object_permission_id": "permission",'
    r' "created_by": "creator", "updated_by": "updater", "created_at": "2026-11-19T10:00:00.123000Z",'
    r' "updated_at": "2026-11-19T10:00:00Z", "litellm_budget_table": {"budget_id": "budget",'
    r' "soft_budget": 8.0, "max_budget": 10.5, "rollover_max_budget": 20.0, "max_parallel_requests": 3,'
    r' "tpm_limit": 1000, "rpm_limit": 10, "tpd_limit": 100000,'
    r' "model_max_budget": {"claude": {"max_budget": 1}}, "model_tpm_limit": {"gpt": 7, "claude": 100},'
    r' "model_rpm_limit": {"claude": 5}, "budget_duration": "30d", "allowed_models": ["claude"],'
    r' "temp_budget_increase": 1.5, "temp_budget_expiry": "2026-11-20T12:30:00.250000Z"},'
    r' "object_permission": null, "last_refreshed_at": null}'
)


def _unstamped(entry: str) -> str:
    """``entry`` with the time it was cached taken out, as the Rust store leaves it for the host that caches the
    project to stamp."""
    return re.sub(r'"last_refreshed_at": [0-9.e+]+', '"last_refreshed_at": null', entry)


_PROJECTS: Final = pytest.mark.parametrize(
    ("row", "entry"),
    [(CREATED_ROW, CREATED_ENTRY), (FULL_ROW, FULL_ENTRY), (BARE_BUDGET_ROW, BARE_BUDGET_ENTRY)],
    ids=["as_created", "every_column_set", "a_budget_with_its_defaults"],
)


@_PROJECTS
def test_the_project_entry_is_what_python_dumps_for_the_repository_row(row: str, entry: str) -> None:
    """The steps ``get_project_object`` takes: the repository's answer decoded as ``LiteLLM_ProjectTable``, dumped
    into ``LiteLLM_ProjectTableCachedObj`` and serialized the way the cache writes it."""
    project: Final = rust_database.decode(LiteLLM_ProjectTable, row)
    assert project is not None
    cached: Final = LiteLLM_ProjectTableCachedObj.model_validate(project.model_dump())

    assert json.dumps(CacheCodec.serialize(cached, model_type=LiteLLM_ProjectTableCachedObj)) == entry


@_PROJECTS
@pytest.mark.asyncio
async def test_the_proxy_caches_a_project_it_read_from_the_database_in_the_bytes_the_rust_store_caches(
    row: str, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores: Final = install_rust_auth_stores(monkeypatch)
    project_id: Final = json.loads(row)["project_id"]
    stores.project.reader[project_id] = LiteLLM_ProjectTable.model_validate_json(row)
    cache: Final = UserApiKeyCache()

    await get_project_object(project_id=project_id, prisma_client=MagicMock(), user_api_key_cache=cache)

    assert _unstamped(json.dumps(cache.get_local(AuthObject.project(project_id)))) == entry


@pytest.mark.parametrize(
    "entry",
    [CREATED_ENTRY, FULL_ENTRY, BARE_BUDGET_ENTRY],
    ids=["as_created", "every_column_set", "a_budget_with_its_defaults"],
)
@pytest.mark.asyncio
async def test_python_caches_the_project_it_read_from_the_entry_in_the_same_bytes(entry: str) -> None:
    """A field added to ``LiteLLM_ProjectTableCachedObj`` or the budget it nests shows up here first: Python then
    writes a field the Rust record does not name, and the Rust side of this contract has to name it too."""
    cache: Final = UserApiKeyCache()
    project_id: Final = json.loads(entry)["project_id"]
    cache.set_local(AuthObject.project(project_id), json.loads(entry), None)

    project: Final = await get_project_object(
        project_id=project_id, prisma_client=MagicMock(), user_api_key_cache=cache
    )

    assert json.dumps(CacheCodec.serialize(project, model_type=LiteLLM_ProjectTableCachedObj)) == entry
