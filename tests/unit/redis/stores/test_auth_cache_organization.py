"""The auth cache entry the Rust auth store shapes for an organization, against Python's own model and its own
organization readers and writers. One ``LiteLLM_OrganizationTable`` dump is the value under every organization key, with
the budget nested only when it was read. The row and entry literals are the ones
``litellm/rust/crates/auth-store/tests/organization.rs`` reads from Postgres and
``litellm/rust/crates/redis-store/tests/auth_cache_organization.rs`` round-trips through Redis. Change all three
together."""

import json
from typing import Final
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.proxy.auth.auth_checks import get_org_object, get_org_object_by_alias, get_org_object_for_request
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from litellm.repositories import rust_database
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

CREATED_ROW: Final = (
    r'{"organization_id":"created","organization_alias":"Created","budget_id":"budget","metadata":{},"models":[],'
    r'"spend":0.0,"model_spend":{},"object_permission_id":null,"created_by":"creator","updated_by":"creator",'
    r'"litellm_budget_table":null}'
)
CREATED_ENTRY: Final = (
    r'{"organization_id": "created", "organization_alias": "Created", "budget_id": "budget", "spend": 0.0,'
    r' "metadata": {}, "models": [], "model_spend": {}, "created_by": "creator", "updated_by": "creator",'
    r' "users": null, "litellm_budget_table": null, "object_permission": null, "object_permission_id": null}'
)
FULL_ROW: Final = (
    r'{"organization_id":"full","organization_alias":"Full","budget_id":"budget","metadata":{"guardrails": ["pii"]},'
    r'"models":["claude","all-proxy-models"],"spend":1.5,"model_spend":{"claude": 1.5},'
    r'"object_permission_id":"permission","created_by":"creator","updated_by":"updater",'
    r'"litellm_budget_table":{"budget_id":"budget","name":"Budget","max_budget":10.5,"rollover_max_budget":20.0,'
    r'"soft_budget":8.0,"max_parallel_requests":3,"tpm_limit":1000,"rpm_limit":10,"tpd_limit":100000,'
    r'"model_max_budget":{"claude": {"max_budget": 1}},"model_tpm_limit":{"gpt": "7", "claude": 100.0},'
    r'"model_rpm_limit":{"claude": "5"},"budget_duration":"30d","budget_limits":[],"cap_policy":"hard",'
    r'"budget_reset_at":"2026-12-01T00:00:00+00:00","budget_reset_count":2,"allowed_models":["claude"],'
    r'"temp_budget_increase":1.5,"temp_budget_expiry":"2026-11-20T12:30:00.250+00:00",'
    r'"created_at":"2026-11-01T00:00:00+00:00","created_by":"test","updated_at":"2026-11-02T00:00:00+00:00",'
    r'"updated_by":"test"}}'
)
FULL_ENTRY: Final = (
    r'{"organization_id": "full", "organization_alias": "Full", "budget_id": "budget", "spend": 1.5,'
    r' "metadata": {"guardrails": ["pii"]}, "models": ["claude", "all-proxy-models"],'
    r' "model_spend": {"claude": 1.5}, "created_by": "creator", "updated_by": "updater", "users": null,'
    r' "litellm_budget_table": {"budget_id": "budget", "soft_budget": 8.0, "max_budget": 10.5,'
    r' "rollover_max_budget": 20.0, "max_parallel_requests": 3, "tpm_limit": 1000, "rpm_limit": 10,'
    r' "tpd_limit": 100000, "model_max_budget": {"claude": {"max_budget": 1}}, "model_tpm_limit": {"gpt": 7,'
    r' "claude": 100}, "model_rpm_limit": {"claude": 5}, "budget_duration": "30d", "allowed_models": ["claude"],'
    r' "temp_budget_increase": 1.5, "temp_budget_expiry": "2026-11-20T12:30:00.250000Z"}, "object_permission": null,'
    r' "object_permission_id": "permission"}'
)
NULLS_ROW: Final = (
    r'{"organization_id":"nulls","organization_alias":"Nulls","budget_id":"bare","metadata":null,"models":[],'
    r'"spend":0.0,"model_spend":null,"object_permission_id":null,"created_by":"creator","updated_by":"creator",'
    r'"litellm_budget_table":{"budget_id":"bare","name":null,"max_budget":null,"soft_budget":null,'
    r'"max_parallel_requests":null,"tpm_limit":null,"rpm_limit":null,"tpd_limit":null,"model_max_budget":null,'
    r'"model_tpm_limit":null,"model_rpm_limit":null,"budget_duration":null,"budget_limits":null,"cap_policy":null,'
    r'"budget_reset_at":null,"budget_reset_count":0,"allowed_models":[],"temp_budget_increase":null,'
    r'"temp_budget_expiry":null,"created_at":"2026-11-01T00:00:00+00:00","created_by":"test",'
    r'"updated_at":"2026-11-01T00:00:00+00:00","updated_by":"test"}}'
)
NULLS_ENTRY: Final = (
    r'{"organization_id": "nulls", "organization_alias": "Nulls", "budget_id": "bare", "spend": 0.0,'
    r' "metadata": null, "models": [], "model_spend": null, "created_by": "creator", "updated_by": "creator",'
    r' "users": null, "litellm_budget_table": {"budget_id": "bare", "soft_budget": null, "max_budget": null,'
    r' "max_parallel_requests": null, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,'
    r' "model_max_budget": null, "model_tpm_limit": null, "model_rpm_limit": null, "budget_duration": null,'
    r' "allowed_models": [], "temp_budget_increase": null, "temp_budget_expiry": null}, "object_permission": null,'
    r' "object_permission_id": null}'
)
FULL_WITHOUT_BUDGET_ROW: Final = (
    r'{"organization_id":"full","organization_alias":"Full","budget_id":"budget","metadata":{"guardrails": ["pii"]},'
    r'"models":["claude","all-proxy-models"],"spend":1.5,"model_spend":{"claude": 1.5},'
    r'"object_permission_id":"permission","created_by":"creator","updated_by":"updater","litellm_budget_table":null}'
)
FULL_WITHOUT_BUDGET_ENTRY: Final = (
    r'{"organization_id": "full", "organization_alias": "Full", "budget_id": "budget", "spend": 1.5,'
    r' "metadata": {"guardrails": ["pii"]}, "models": ["claude", "all-proxy-models"],'
    r' "model_spend": {"claude": 1.5}, "created_by": "creator", "updated_by": "updater", "users": null,'
    r' "litellm_budget_table": null, "object_permission": null, "object_permission_id": "permission"}'
)

_FIELDS: Final = TypeAdapter(dict[str, object])
_FULL: Final = json.loads(FULL_ROW)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata", [1]),
        ("metadata", "{}"),
        ("model_spend", [1]),
        (
            "litellm_budget_table",
            {**_FULL["litellm_budget_table"], "budget_limits": [{"budget_duration": "1d", "max_budget": 1}]},
        ),
    ],
    ids=[
        "metadata_that_is_a_list",
        "metadata_text_the_model_does_not_parse",
        "model_spend_that_is_a_list",
        "a_budget_with_concurrent_windows",
    ],
)
def test_python_refuses_each_row_the_rust_store_refuses(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        LiteLLM_OrganizationTable.model_validate({**_FULL, field: value})


@pytest.mark.parametrize(
    ("row", "entry"),
    [
        (CREATED_ROW, CREATED_ENTRY),
        (FULL_ROW, FULL_ENTRY),
        (FULL_WITHOUT_BUDGET_ROW, FULL_WITHOUT_BUDGET_ENTRY),
        (NULLS_ROW, NULLS_ENTRY),
    ],
    ids=[
        "as_created",
        "every_column_set_with_its_budget",
        "every_column_set_without_its_budget",
        "json_nulls_and_a_bare_budget",
    ],
)
def test_the_organization_entry_is_what_python_dumps_for_the_repository_row(row: str, entry: str) -> None:
    organization: Final = rust_database.decode(LiteLLM_OrganizationTable, row)
    assert organization is not None

    assert json.dumps(CacheCodec.serialize(organization, model_type=LiteLLM_OrganizationTable)) == entry


@pytest.mark.parametrize(
    ("row", "include_budget_table", "cache_key", "entry"),
    [
        (CREATED_ROW, False, AuthObject.org("created"), CREATED_ENTRY),
        (FULL_ROW, True, AuthObject.org_with_budget("full"), FULL_ENTRY),
        (FULL_ROW, False, AuthObject.org("full"), FULL_WITHOUT_BUDGET_ENTRY),
        (NULLS_ROW, True, AuthObject.org_with_budget("nulls"), NULLS_ENTRY),
    ],
    ids=[
        "as_created",
        "every_column_set_with_its_budget",
        "every_column_set_without_its_budget",
        "json_nulls_and_a_bare_budget",
    ],
)
@pytest.mark.asyncio
async def test_the_proxy_caches_an_organization_it_read_by_id_in_the_bytes_the_rust_store_caches(
    row: str, include_budget_table: bool, cache_key: AuthObject, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores: Final = install_rust_auth_stores(monkeypatch)
    organization_id: Final = json.loads(row)["organization_id"]
    stores.organization.rows[organization_id] = _FIELDS.validate_json(row)
    cache: Final = UserApiKeyCache()

    await get_org_object(
        org_id=organization_id,
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        include_budget_table=include_budget_table,
    )

    assert json.dumps(cache.get_local(cache_key)) == entry


@pytest.mark.asyncio
async def test_the_proxy_keeps_the_organization_a_request_read_as_its_last_known_entry_in_the_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores: Final = install_rust_auth_stores(monkeypatch)
    stores.organization.rows["full"] = _FIELDS.validate_json(FULL_ROW)
    cache: Final = UserApiKeyCache()

    await get_org_object_for_request(
        org_id="full",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    assert json.dumps(cache.get_local(AuthObject.org_with_budget_last_known("full"))) == FULL_ENTRY


@pytest.mark.asyncio
async def test_the_proxy_caches_an_organization_it_read_by_alias_under_the_alias_and_the_id_in_the_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores: Final = install_rust_auth_stores(monkeypatch)
    stores.organization.rows["full"] = _FIELDS.validate_json(FULL_ROW)
    cache: Final = UserApiKeyCache()

    await get_org_object_by_alias(org_alias="Full", prisma_client=MagicMock(), user_api_key_cache=cache)

    assert (
        json.dumps(cache.get_local(AuthObject.org_alias("Full"))),
        json.dumps(cache.get_local(AuthObject.org("full"))),
    ) == (FULL_WITHOUT_BUDGET_ENTRY, FULL_WITHOUT_BUDGET_ENTRY)


@pytest.mark.parametrize(
    ("cache_key", "include_budget_table", "entry"),
    [
        (AuthObject.org("created"), False, CREATED_ENTRY),
        (AuthObject.org_with_budget("full"), True, FULL_ENTRY),
        (AuthObject.org("full"), False, FULL_WITHOUT_BUDGET_ENTRY),
        (AuthObject.org_with_budget("nulls"), True, NULLS_ENTRY),
    ],
    ids=[
        "as_created",
        "every_column_set_with_its_budget",
        "every_column_set_without_its_budget",
        "json_nulls_and_a_bare_budget",
    ],
)
@pytest.mark.asyncio
async def test_python_caches_the_organization_it_read_from_the_entry_in_the_same_bytes(
    cache_key: AuthObject, include_budget_table: bool, entry: str
) -> None:
    """A field added to ``LiteLLM_OrganizationTable`` or the budget it nests shows up here first: Python then writes a
    field the Rust record does not name, and the Rust side of this contract has to name it too."""
    cache: Final = UserApiKeyCache()
    cache.set_local(cache_key, _FIELDS.validate_json(entry), None)

    organization: Final = await get_org_object(
        org_id=json.loads(entry)["organization_id"],
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        include_budget_table=include_budget_table,
    )

    assert json.dumps(CacheCodec.serialize(organization, model_type=LiteLLM_OrganizationTable)) == entry
