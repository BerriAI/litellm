"""The auth cache entry the Rust auth store shapes for a user, against Python's own model and its own user reader and
writer. The entry sits under the bare user id. The row and entry literals are the ones
``litellm/rust/crates/auth-store/tests/user.rs`` reads from Postgres and
``litellm/rust/crates/redis-store/tests/auth_cache_user.rs`` round-trips through Redis. Change all three together."""

import json
from typing import Final
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.caching.dual_cache import LimitedSizeOrderedDict
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth.auth_checks import get_user_object
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from litellm.repositories import rust_database
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

CREATED_ROW: Final = (
    r'{"user_id":"created","user_alias":null,"team_id":null,"sso_user_id":null,"organization_id":null,'
    r'"object_permission_id":null,"password_reset_required":null,"last_breach_check_at":null,"teams":[],'
    r'"user_role":null,"max_budget":null,"budget_limits":null,"spend":0.0,"user_email":null,"models":[],'
    r'"metadata":{},"max_parallel_requests":null,"tpm_limit":null,"rpm_limit":null,"budget_duration":null,'
    r'"budget_reset_at":null,"allowed_cache_controls":[],"policies":[],"model_spend":{},"model_max_budget":{},'
    r'"created_at":"2026-11-01T00:00:00+00:00","updated_at":"2026-11-01T00:00:00+00:00",'
    r'"organization_memberships":[]}'
)
CREATED_ENTRY: Final = (
    r'{"user_id": "created", "user_alias": null, "team_id": null, "sso_user_id": null, "organization_id": null,'
    r' "object_permission_id": null, "password_reset_required": null, "last_breach_check_at": null, "teams": [],'
    r' "user_role": null, "max_budget": null, "budget_limits": null, "spend": 0.0, "user_email": null, "models": [],'
    r' "metadata": {}, "max_parallel_requests": null, "tpm_limit": null, "rpm_limit": null, "budget_duration": null,'
    r' "budget_reset_at": null, "allowed_cache_controls": [], "policies": [], "model_spend": {},'
    r' "model_max_budget": {}, "created_at": "2026-11-01T00:00:00Z", "updated_at": "2026-11-01T00:00:00Z",'
    r' "organization_memberships": [], "object_permission": null}'
)
NULLS_ROW: Final = (
    r'{"user_id":"nulls","user_alias":null,"team_id":null,"sso_user_id":null,"organization_id":null,'
    r'"object_permission_id":null,"password_reset_required":null,"last_breach_check_at":null,"teams":[],'
    r'"user_role":null,"max_budget":null,"budget_limits":null,"spend":0.0,"user_email":null,"models":[],'
    r'"metadata":null,"max_parallel_requests":null,"tpm_limit":null,"rpm_limit":null,"budget_duration":null,'
    r'"budget_reset_at":null,"allowed_cache_controls":[],"policies":[],"model_spend":null,"model_max_budget":null,'
    r'"created_at":"2026-11-01T00:00:00+00:00","updated_at":"2026-11-01T00:00:00+00:00",'
    r'"organization_memberships":[]}'
)
NULLS_ENTRY: Final = (
    r'{"user_id": "nulls", "user_alias": null, "team_id": null, "sso_user_id": null, "organization_id": null,'
    r' "object_permission_id": null, "password_reset_required": null, "last_breach_check_at": null, "teams": [],'
    r' "user_role": null, "max_budget": null, "budget_limits": null, "spend": 0.0, "user_email": null, "models": [],'
    r' "metadata": null, "max_parallel_requests": null, "tpm_limit": null, "rpm_limit": null,'
    r' "budget_duration": null, "budget_reset_at": null, "allowed_cache_controls": [], "policies": [],'
    r' "model_spend": null, "model_max_budget": null, "created_at": "2026-11-01T00:00:00Z",'
    r' "updated_at": "2026-11-01T00:00:00Z", "organization_memberships": [], "object_permission": null}'
)
FULL_ROW: Final = (
    r'{"user_id":"full","user_alias":"Full","team_id":"team","sso_user_id":"sso","organization_id":"org-a",'
    r'"object_permission_id":"permission","password_reset_required":true,'
    r'"last_breach_check_at":"2026-11-20T12:30:00.250+00:00","teams":["team","other-team"],'
    r'"user_role":"internal_user","max_budget":10.5,"rollover_max_budget":20.0,"rollover_credit":2.5,'
    r'"budget_limits":[{"later": 1, "reset_at": "2026-12-01T00:00:00+00:00", "max_budget": 5,'
    r' "window_start": "2026-11-30T00:00:00+00:00", "budget_duration": "1d"}],"spend":1.5,'
    r'"user_email":"full@example.com","models":["claude","gpt"],"metadata":{"guardrails": ["pii"]},'
    r'"max_parallel_requests":3,"tpm_limit":1000,"rpm_limit":10,"budget_duration":"30d",'
    r'"budget_reset_at":"2026-12-01T00:00:00+00:00","allowed_cache_controls":["no-cache"],"policies":["policy"],'
    r'"model_spend":{"claude": 1.5},"model_max_budget":{"claude": {"max_budget": 1}},'
    r'"created_at":"2026-11-01T00:00:00+00:00","updated_at":"2026-11-02T00:00:00+00:00",'
    r'"organization_memberships":[{"user_id":"full","organization_id":"org-a","user_role":null,"spend":0.0,'
    r'"budget_id":null,"created_at":"2026-11-03T00:00:00+00:00",'
    r'"updated_at":"2026-11-03T00:00:00+00:00"},{"user_id":"full","organization_id":"org-b","user_role":"org_admin",'
    r'"spend":0.5,"budget_id":"budget","created_at":"2026-11-03T00:00:00+00:00",'
    r'"updated_at":"2026-11-04T00:00:00.500+00:00"}]}'
)
FULL_ENTRY: Final = (
    r'{"user_id": "full", "user_alias": "Full", "team_id": "team", "sso_user_id": "sso", "organization_id": "org-a",'
    r' "object_permission_id": "permission", "password_reset_required": true,'
    r' "last_breach_check_at": "2026-11-20T12:30:00.250000Z", "teams": ["team", "other-team"],'
    r' "user_role": "internal_user", "max_budget": 10.5, "rollover_max_budget": 20.0, "rollover_credit": 2.5,'
    r' "budget_limits": [{"budget_duration": "1d", "max_budget": 5.0, "reset_at": "2026-12-01T00:00:00Z",'
    r' "window_start": "2026-11-30T00:00:00Z"}], "spend": 1.5, "user_email": "full@example.com",'
    r' "models": ["claude", "gpt"], "metadata": {"guardrails": ["pii"]}, "max_parallel_requests": 3,'
    r' "tpm_limit": 1000, "rpm_limit": 10, "budget_duration": "30d", "budget_reset_at": "2026-12-01T00:00:00Z",'
    r' "allowed_cache_controls": ["no-cache"], "policies": ["policy"], "model_spend": {"claude": 1.5},'
    r' "model_max_budget": {"claude": {"max_budget": 1}}, "created_at": "2026-11-01T00:00:00Z",'
    r' "updated_at": "2026-11-02T00:00:00Z", "organization_memberships": [{"user_id": "full",'
    r' "organization_id": "org-a", "user_role": null, "spend": 0.0, "budget_id": null,'
    r' "created_at": "2026-11-03T00:00:00Z", "updated_at": "2026-11-03T00:00:00Z", "user": null,'
    r' "litellm_budget_table": null, "user_email": null}, {"user_id": "full", "organization_id": "org-b",'
    r' "user_role": "org_admin", "spend": 0.5, "budget_id": "budget", "created_at": "2026-11-03T00:00:00Z",'
    r' "updated_at": "2026-11-04T00:00:00.500000Z", "user": null, "litellm_budget_table": null,'
    r' "user_email": null}], "object_permission": null}'
)

_FIELDS: Final = TypeAdapter(dict[str, object])
_CASES: Final = [(CREATED_ROW, CREATED_ENTRY), (FULL_ROW, FULL_ENTRY), (NULLS_ROW, NULLS_ENTRY)]
_IDS: Final = ["as_created", "every_column_set", "json_nulls"]
_FULL: Final = json.loads(FULL_ROW)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata", [1]),
        ("metadata", "{}"),
        ("model_spend", [1]),
        ("model_max_budget", "{}"),
        ("budget_limits", [{"budget_duration": "1d"}]),
        ("organization_memberships", [{**_FULL["organization_memberships"][0], "spend": None}]),
        ("organization_memberships", [{**_FULL["organization_memberships"][0], "created_at": None}]),
        ("organization_memberships", [{**_FULL["organization_memberships"][0], "updated_at": None}]),
    ],
    ids=[
        "metadata_that_is_a_list",
        "metadata_text_the_model_does_not_parse",
        "model_spend_that_is_a_list",
        "model_max_budget_that_is_text",
        "a_budget_window_without_its_max_budget",
        "a_membership_without_its_spend",
        "a_membership_without_its_creation_time",
        "a_membership_without_its_update_time",
    ],
)
def test_python_refuses_each_row_the_rust_store_refuses(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        LiteLLM_UserTable.model_validate({**_FULL, field: value})


@pytest.mark.parametrize(("row", "entry"), _CASES, ids=_IDS)
def test_the_user_entry_is_what_python_dumps_for_the_repository_row(row: str, entry: str) -> None:
    user: Final = rust_database.decode(LiteLLM_UserTable, row)
    assert user is not None

    assert json.dumps(CacheCodec.serialize(user, model_type=LiteLLM_UserTable)) == entry


@pytest.mark.parametrize(("row", "entry"), _CASES, ids=_IDS)
@pytest.mark.parametrize("user_id_upsert", [False, True], ids=["read", "upserted"])
@pytest.mark.asyncio
async def test_the_proxy_caches_a_user_it_read_or_upserted_in_the_bytes_the_rust_store_caches(
    row: str, entry: str, user_id_upsert: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upserted, the reader has no row and the insert answers with the row a concurrent sign-in stored."""
    monkeypatch.setattr(auth_checks, "last_db_access_time", LimitedSizeOrderedDict(max_size=100))
    stores: Final = install_rust_auth_stores(monkeypatch)
    user_id: Final = json.loads(row)["user_id"]
    rows: Final = {user_id: _FIELDS.validate_json(row)}
    if user_id_upsert:
        stores.user.writer = rows
    else:
        stores.user.rows = rows
    cache: Final = UserApiKeyCache()

    await get_user_object(
        user_id=user_id, prisma_client=MagicMock(), user_api_key_cache=cache, user_id_upsert=user_id_upsert
    )

    assert json.dumps(cache.get_local(AuthObject.user(user_id))) == entry


@pytest.mark.parametrize("entry", [CREATED_ENTRY, FULL_ENTRY, NULLS_ENTRY], ids=_IDS)
@pytest.mark.asyncio
async def test_python_caches_the_user_it_read_from_the_entry_in_the_same_bytes(entry: str) -> None:
    """A field added to ``LiteLLM_UserTable`` or the membership it nests shows up here first: Python then writes a
    field the Rust record does not name, and the Rust side of this contract has to name it too."""
    user_id: Final = json.loads(entry)["user_id"]
    cache: Final = UserApiKeyCache()
    cache.set_local(AuthObject.user(user_id), _FIELDS.validate_json(entry), None)

    user: Final = await get_user_object(
        user_id=user_id, prisma_client=MagicMock(), user_api_key_cache=cache, user_id_upsert=False
    )

    assert json.dumps(CacheCodec.serialize(user, model_type=LiteLLM_UserTable)) == entry
