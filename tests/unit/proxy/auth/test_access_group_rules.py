from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from litellm.proxy._types import ConfigGeneralSettings, LiteLLM_AccessGroupTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.access_group_rules import AccessGroupRule, access_group_ids_for_identity
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


@dataclass(frozen=True)
class _Database:
    db: object


@pytest.mark.parametrize(
    "claims,expected",
    (
        ({"groups": ["science"], "tenant": {"id": "production"}}, ("science-id", "standard-id")),
        ({"groups": ["other"]}, ()),
        ({"groups": "science", "tenant": {"id": ["production"]}}, ()),
        ({}, ()),
        ({"groups": ["science", "science"]}, ("science-id",)),
        ({"groups": [{"name": "science"}]}, ()),
    ),
)
@pytest.mark.asyncio
async def test_verified_attributes_grant_union_with_strict_claim_shapes(
    claims: Mapping[str, object], expected: tuple[str, ...]
) -> None:
    rules: Final = (
        AccessGroupRule(attribute="jwt_claims.groups", contains="science", access_group="science"),
        AccessGroupRule(attribute="jwt_claims.tenant.id", equals="production", access_group="standard"),
    )
    cache: Final = UserApiKeyCache()
    for name in ("science", "standard"):
        group: Final = LiteLLM_AccessGroupTable(access_group_id=f"{name}-id", access_group_name=name)
        cache.set_cache(f"access_group_rule_name:{name}", group, model_type=LiteLLM_AccessGroupTable)
        cache.set_cache(f"access_group_id:{name}-id", group, model_type=LiteLLM_AccessGroupTable)
    assert (
        await access_group_ids_for_identity(
            UserAPIKeyAuth(jwt_claims=dict(claims)), rules=rules, cache=cache, prisma_client=_Database(object())
        )
        == expected
    )


@pytest.mark.parametrize("rule", ({"contains": "science", "equals": "science"}, {}))
def test_invalid_rule_fails_general_settings_validation(rule: Mapping[str, str]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ConfigGeneralSettings.model_validate(
            {"access_group_rules": [{"attribute": "jwt_claims.groups", "access_group": "science", **rule}]}
        )


@pytest.mark.parametrize("attribute", ("metadata.department", "user_email", "user_email.domain", "team_id", "org_id"))
def test_caller_editable_attributes_are_rejected_as_grant_sources(attribute: str) -> None:
    with pytest.raises(ValidationError, match="administrator- or IdP-controlled"):
        ConfigGeneralSettings.model_validate(
            {"access_group_rules": [{"attribute": attribute, "equals": "science", "access_group": "science"}]}
        )


@pytest.mark.asyncio
async def test_missing_group_and_no_database_grant_nothing() -> None:
    rules: Final = (AccessGroupRule(attribute="user_id", equals="subject", access_group="absent"),)
    table: Final = MagicMock()
    table.find_unique = AsyncMock(return_value=None)
    cache: Final = UserApiKeyCache()
    assert (
        await access_group_ids_for_identity(
            UserAPIKeyAuth(user_id="subject"),
            rules=rules,
            cache=cache,
            prisma_client=_Database(MagicMock(litellm_accessgrouptable=table)),
        )
        == ()
    )
    assert await access_group_ids_for_identity(UserAPIKeyAuth(user_id="subject"), rules=rules, cache=cache) == ()
    assert await access_group_ids_for_identity(
        UserAPIKeyAuth(user_id="subject"), rules=rules, prisma_client=_Database(object())
    ) == ()


@pytest.mark.asyncio
async def test_database_failure_does_not_turn_into_an_unrestricted_grant() -> None:
    table: Final = MagicMock()
    table.find_unique = AsyncMock(side_effect=RuntimeError("database unavailable"))
    with pytest.raises(RuntimeError, match="database unavailable"):
        await access_group_ids_for_identity(
            UserAPIKeyAuth(user_id="subject"),
            rules=(AccessGroupRule(attribute="user_id", equals="subject", access_group="science"),),
            cache=UserApiKeyCache(),
            prisma_client=_Database(MagicMock(litellm_accessgrouptable=table)),
        )


@pytest.mark.asyncio
async def test_virtual_key_user_grants_are_additive_and_do_not_mutate_cached_identity() -> None:
    identity: Final = UserAPIKeyAuth(models=["explicit"], user_id="science", access_group_ids=["existing"])
    group: Final = LiteLLM_AccessGroupTable(
        access_group_id="science-id", access_group_name="science", access_model_names=["group-model"]
    )
    cache: Final = UserApiKeyCache()
    cache.set_cache("access_group_rule_name:science", group, model_type=LiteLLM_AccessGroupTable)
    cache.set_cache("access_group_id:science-id", group, model_type=LiteLLM_AccessGroupTable)
    ids: Final = await access_group_ids_for_identity(
        identity,
        rules=(AccessGroupRule(attribute="user_id", equals="science", access_group="science"),),
        cache=cache,
        prisma_client=_Database(object()),
    )
    granted: Final = identity.model_copy()
    granted.apply_access_group_rule_grants(ids)
    assert granted.access_group_ids == ["existing", "science-id"]
    assert granted.access_group_rule_grants == frozenset({"science-id"})
    assert identity.access_group_ids == ["existing"]
    identity.apply_access_group_rule_grants(())
    assert identity.access_group_ids == ["existing"]


@pytest.mark.asyncio
async def test_null_equals_does_not_match_an_absent_claim() -> None:
    rules: Final = (AccessGroupRule(attribute="jwt_claims.absent", equals=None, access_group="science"),)
    assert (
        await access_group_ids_for_identity(UserAPIKeyAuth(jwt_claims={}), rules=rules, cache=UserApiKeyCache()) == ()
    )


@pytest.mark.parametrize(
    "claims,local_email,expected",
    (
        ({"email": "science@example.com"}, "other@example.com", ("science-id",)),
        ({"email": "other@example.com"}, "science@example.com", ()),
        ({}, "science@example.com", ()),
        ({"email": ["science@example.com"]}, "science@example.com", ()),
    ),
)
@pytest.mark.asyncio
async def test_email_grants_read_verified_claims_not_the_locally_editable_email(
    claims: Mapping[str, object], local_email: str, expected: tuple[str, ...]
) -> None:
    group: Final = LiteLLM_AccessGroupTable(access_group_id="science-id", access_group_name="science")
    cache: Final = UserApiKeyCache()
    cache.set_cache("access_group_rule_name:science", group, model_type=LiteLLM_AccessGroupTable)
    cache.set_cache("access_group_id:science-id", group, model_type=LiteLLM_AccessGroupTable)
    assert (
        await access_group_ids_for_identity(
            UserAPIKeyAuth(jwt_claims=dict(claims), user_email=local_email),
            rules=(
                AccessGroupRule(attribute="jwt_claims.email", equals="science@example.com", access_group="science"),
            ),
            cache=cache,
            prisma_client=_Database(object()),
        )
        == expected
    )


@pytest.mark.parametrize("operator", ("contains", "equals"))
@pytest.mark.asyncio
async def test_null_operator_remains_distinct_when_rules_are_reloaded(operator: str) -> None:
    table: Final = MagicMock()
    table.find_unique = AsyncMock(return_value={"access_group_id": "science-id", "access_group_name": "science"})
    cache: Final = UserApiKeyCache()
    database: Final = _Database(MagicMock(litellm_accessgrouptable=table))
    rule: Final = AccessGroupRule.model_validate(
        {"attribute": "jwt_claims.value", operator: None, "access_group": "science"}
    )
    principal: Final = UserAPIKeyAuth(jwt_claims={"value": [None] if operator == "contains" else None})
    assert await access_group_ids_for_identity(principal, rules=(rule,), cache=cache, prisma_client=database) == (
        "science-id",
    )


@pytest.mark.asyncio
async def test_authenticated_identity_fields_and_claims_are_available_without_metadata_flattening() -> None:
    table: Final = MagicMock()
    table.find_unique = AsyncMock(return_value={"access_group_id": "science-id", "access_group_name": "science"})
    identity: Final = UserAPIKeyAuth(
        user_id="verified-user", user_role=LitellmUserRoles.INTERNAL_USER, jwt_claims={"groups": ["science"]}
    )
    rules: Final = tuple(
        AccessGroupRule.model_validate({"attribute": attribute, operator: value, "access_group": "science"})
        for attribute, operator, value in (
            ("user_id", "equals", "verified-user"),
            ("jwt_claims.groups", "contains", "science"),
            ("user_role", "equals", "internal_user"),
        )
    )
    database: Final = _Database(MagicMock(litellm_accessgrouptable=table))
    for index, expected in ((0, ("science-id",)), (1, ("science-id",)), (2, ("science-id",))):
        assert (
            await access_group_ids_for_identity(
                identity, rules=(rules[index],), cache=UserApiKeyCache(), prisma_client=database
            )
            == expected
        )


@pytest.mark.asyncio
async def test_name_resolution_is_cached_and_renamed_group_does_not_keep_granting_old_name() -> None:
    table: Final = MagicMock()
    table.find_unique = AsyncMock(return_value={"access_group_id": "science-id", "access_group_name": "science"})
    database: Final = _Database(MagicMock(litellm_accessgrouptable=table))
    cache: Final = UserApiKeyCache()
    rules: Final = (AccessGroupRule(attribute="user_id", equals="subject", access_group="science"),)
    assert await access_group_ids_for_identity(
        UserAPIKeyAuth(user_id="subject"), rules=rules, cache=cache, prisma_client=database
    ) == ("science-id",)
    cache.set_cache(
        "access_group_id:science-id",
        LiteLLM_AccessGroupTable(access_group_id="science-id", access_group_name="science"),
        model_type=LiteLLM_AccessGroupTable,
    )
    assert await access_group_ids_for_identity(
        UserAPIKeyAuth(user_id="subject"), rules=rules, cache=cache, prisma_client=database
    ) == ("science-id",)
    assert table.find_unique.await_count == 1
    cache.set_cache(
        "access_group_id:science-id",
        LiteLLM_AccessGroupTable(access_group_id="science-id", access_group_name="renamed"),
        model_type=LiteLLM_AccessGroupTable,
    )
    table.find_unique.return_value = None
    assert (
        await access_group_ids_for_identity(
            UserAPIKeyAuth(user_id="subject"), rules=rules, cache=cache, prisma_client=database
        )
        == ()
    )


@pytest.mark.parametrize(
    "attribute",
    ("groups", "department", "metadata.department", "models", "access_group_rule_grants", "jwt_claims..groups"),
)
def test_attribute_requires_a_real_principal_field(attribute: str) -> None:
    with pytest.raises(ValidationError, match="attribute"):
        ConfigGeneralSettings.model_validate(
            {"access_group_rules": [{"attribute": attribute, "contains": "science", "access_group": "science"}]}
        )
