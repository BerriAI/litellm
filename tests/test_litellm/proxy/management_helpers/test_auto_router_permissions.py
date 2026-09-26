from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.management_helpers.auto_router_permissions import (
    MemberAutoRouterDependencyObjects,
    authorize_member_auto_router_dependencies,
    authorize_member_auto_router_team,
    authorize_member_auto_router_write,
    validate_member_auto_router_config,
)
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo, updateDeployment


class _ReadTable:
    async def find_unique(self, where: Mapping[str, object], include: Mapping[str, object] | None = None) -> None:
        return None


@dataclass(frozen=True)
class _PermissionDb:
    litellm_teammembership: _ReadTable = _ReadTable()


@dataclass(frozen=True)
class _Client:
    db: _PermissionDb = _PermissionDb()


def _team(**updates: object) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable.model_validate(
        {
            "team_id": "team-a",
            "models": ["allowed"],
            "members_with_roles": [Member(user_id="owner", role="user")],
            "team_member_permissions": ["/auto_router/manage"],
            **updates,
        }
    )


def _actor(**updates: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth.model_validate(
        {"user_id": "owner", "user_role": "internal_user", "models": ["allowed"], **updates}
    )


@pytest.fixture
def catalog() -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"}}
            for name in ("allowed", "other")
        ]
    )


@pytest.mark.parametrize(
    "actor_updates,team_updates,premium,allowed",
    [
        ({}, {}, True, True),
        ({"team_id": UI_TEAM_ID}, {}, True, True),
        ({"team_id": "team-a"}, {}, True, True),
        ({"user_role": LitellmUserRoles.TEAM}, {}, True, True),
        ({"user_role": LitellmUserRoles.ORG_ADMIN}, {}, True, True),
        ({"team_id": "team-b"}, {}, True, False),
        ({"user_id": None}, {}, True, False),
        ({"user_id": ""}, {}, True, False),
        ({"user_id": "peer"}, {}, True, False),
        ({"user_role": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY}, {}, True, False),
        ({"user_role": LitellmUserRoles.CUSTOMER}, {}, True, False),
        ({}, {"team_member_permissions": []}, True, False),
        ({}, {"team_member_permissions": None}, True, False),
        ({}, {"blocked": True}, True, False),
        ({}, {}, False, False),
    ],
)
def test_opt_in_requires_live_named_membership_and_write_role(
    actor_updates: Mapping[str, object], team_updates: Mapping[str, object], premium: bool, allowed: bool
) -> None:
    if allowed:
        authorize_member_auto_router_team(
            user_api_key_dict=_actor(**actor_updates), team=_team(**team_updates), premium_user=premium
        )
        return
    with pytest.raises(HTTPException) as denied:
        authorize_member_auto_router_team(
            user_api_key_dict=_actor(**actor_updates), team=_team(**team_updates), premium_user=premium
        )
    assert denied.value.status_code == 403


@pytest.mark.parametrize("placement", ["inline", "normalized"])
@pytest.mark.parametrize(
    "overrides", [{"api_base": "https://example.invalid"}, {"api_key": "fake"}, {"metadata": {}}, {"model": "other"}]
)
def test_all_tier_parameter_representations_reject_privileged_overrides(
    placement: str, overrides: Mapping[str, object]
) -> None:
    entry: Final = {"model_name": "allowed", "litellm_params": overrides}
    config: Final = (
        {"tiers": {"SIMPLE": [entry]}}
        if placement == "inline"
        else {"tiers": {"SIMPLE": ["allowed"]}, "tier_model_configs": {"SIMPLE": [entry]}}
    )
    with pytest.raises(HTTPException) as denied:
        validate_member_auto_router_config(config)
    assert denied.value.status_code == 400


def test_tier_config_is_normalized_and_unknown_router_extras_are_rejected() -> None:
    validated: Final = validate_member_auto_router_config(
        {"tiers": {"SIMPLE": [{"model_name": "allowed", "litellm_params": {"reasoning_effort": "low"}}]}}
    )
    assert validated.tiers == {"SIMPLE": ["allowed"]}
    assert validated.tier_model_configs["SIMPLE"][0].litellm_params == {"reasoning_effort": "low"}
    assert validate_member_auto_router_config(validated.model_dump()).tiers == validated.tiers
    with pytest.raises(HTTPException):
        validate_member_auto_router_config({"tiers": {"SIMPLE": "allowed"}, "api_base": "https://example.invalid"})


@pytest.mark.parametrize(
    ("jev_override", "rejected_at"),
    [
        ({"api_base": "https://collector.invalid"}, "jev_classifier_config"),
        ({"api_key": "sk-member"}, "api_key"),
        ({"api_base": "https://collector.invalid", "api_key": "sk-member"}, "api_key"),
        ({"api_base": "https://collector.invalid", "api_key": ""}, "jev_classifier_config.api_key"),
        ({"provider": "laya", "laya_api_base": "https://collector.invalid"}, "laya_api_base"),
        ({"provider": "laya", "laya_api_key": "sk-member"}, "laya_api_key"),
    ],
)
def test_members_cannot_move_the_jev_classifier_off_the_proxys_typesafe_account(
    jev_override: Mapping[str, str], rejected_at: str
) -> None:
    with pytest.raises(HTTPException) as denied:
        validate_member_auto_router_config(
            {"tiers": {"SIMPLE": "allowed"}, "classifier_type": "jev", "jev_classifier_config": jev_override}
        )
    assert denied.value.status_code == 400
    assert denied.value.detail == f"Invalid member auto-router configuration at {rejected_at}."


@pytest.mark.parametrize(("provider", "model"), (("typesafe", "jev-preview"), ("laya", "english")))
def test_members_can_still_tune_the_jev_classifier(provider: str, model: str) -> None:
    validated: Final = validate_member_auto_router_config(
        {
            "tiers": {"SIMPLE": "allowed"},
            "classifier_type": "jev",
            "jev_classifier_config": {"provider": provider, "model": model, "timeout_ms": 500},
        }
    )
    assert validated.jev_classifier_config is not None
    assert validated.jev_classifier_config.provider == provider
    assert (validated.jev_classifier_config.model, validated.jev_classifier_config.timeout_ms) == (model, 500)
    assert validate_member_auto_router_config(validated.model_dump()).jev_classifier_config is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch_fields",
    [
        {},
        {"model_name": "renamed"},
        {"blocked": False},
        {"model_info": {"team_id": "other-team"}},
        {"model_info": {"member_auto_router": False}},
        {"litellm_params": {"model": "auto_router/quality_router"}},
        {"litellm_params": {"api_key": "fake"}},
    ],
)
async def test_member_updates_restrict_fields_and_preserve_an_inherited_default(
    catalog: Router, monkeypatch: pytest.MonkeyPatch, patch_fields: Mapping[str, object]
) -> None:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    monkeypatch.setenv("LITELLM_SALT_KEY", "member-router-test-salt")
    existing: Final = Deployment(
        model_name="model_name_team-a_uuid",
        litellm_params=LiteLLM_Params(
            model=encrypt_value_helper("auto_router/complexity_router"),
            complexity_router_config={"tiers": {"SIMPLE": "allowed"}},
            complexity_router_default_model=encrypt_value_helper("allowed"),
        ),
        model_info=ModelInfo(id="router-a", team_id="team-a", team_public_model_name="my-router"),
        created_by="owner",
    )
    patch: Final = updateDeployment.model_validate(
        {"litellm_params": {"complexity_router_config": {"tiers": {"SIMPLE": "allowed"}}}, **patch_fields}
    )
    operation: Final = authorize_member_auto_router_write(
        incoming=patch,
        existing=existing,
        user_api_key_dict=_actor(),
        team=_team(),
        premium_user=True,
        prisma_client=_Client(),
        llm_router=catalog,
    )
    if patch_fields:
        with pytest.raises(HTTPException) as denied:
            await operation
        assert denied.value.status_code == 403
        return
    granted: Final = await operation
    assert granted.default_model == "allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["missing", "nested"])
async def test_member_dependencies_require_plain_configured_models(target: str) -> None:
    catalog: Final = Router(
        model_list=[
            {"model_name": "allowed", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"}},
            {
                "model_name": "nested",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "allowed"}},
                },
            },
        ]
    )
    with pytest.raises(HTTPException) as denied:
        await authorize_member_auto_router_dependencies(
            config=validate_member_auto_router_config({"tiers": {"SIMPLE": target}}),
            default_model=None,
            user_api_key_dict=_actor(models=[target]),
            team=_team(models=[target]),
            prisma_client=_Client(),
            llm_router=catalog,
        )
    assert denied.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("restricted", ["key", "team", None])
async def test_jev_evaluation_requires_model_access_but_no_completion_deployment(
    catalog: Router, restricted: str | None
) -> None:
    permitted: Final = ["allowed", "typesafe/jev-latest"]
    operation: Final = authorize_member_auto_router_dependencies(
        config=validate_member_auto_router_config(
            {"tiers": {"SIMPLE": "allowed"}, "classifier_type": "jev", "jev_classifier_config": {}}
        ),
        default_model=None,
        user_api_key_dict=_actor(models=["allowed"] if restricted == "key" else permitted),
        team=_team(models=["allowed"] if restricted == "team" else permitted),
        prisma_client=_Client(),
        llm_router=catalog,
    )
    if restricted is not None:
        with pytest.raises(ProxyException, match="jev-latest"):
            await operation
        return
    await operation
    assert not catalog.get_model_list("typesafe/jev-latest")


@pytest.mark.asyncio
@pytest.mark.parametrize("restricted", ["member", "project", "organization", None])
async def test_jev_evaluation_obeys_each_containing_scope(catalog: Router, restricted: str | None) -> None:
    allowed: Final = ["allowed", "typesafe/jev-latest"]
    membership: Final = LiteLLM_TeamMembership.model_validate(
        {
            "user_id": "owner",
            "team_id": "team-a",
            "litellm_budget_table": {"allowed_models": ["allowed"] if restricted == "member" else allowed},
        }
    )
    organization: Final = LiteLLM_OrganizationTable.model_validate(
        {
            "organization_id": "org-a",
            "models": ["allowed"] if restricted == "organization" else allowed,
            "budget_id": "org-budget",
            "created_by": "admin",
            "updated_by": "admin",
        }
    )
    project: Final = LiteLLM_ProjectTable.model_validate(
        {"project_id": "project-a", "team_id": "team-a", "models": ["allowed"] if restricted == "project" else allowed}
    )
    operation: Final = authorize_member_auto_router_dependencies(
        config=validate_member_auto_router_config(
            {"tiers": {"SIMPLE": "allowed"}, "classifier_type": "jev", "jev_classifier_config": {}}
        ),
        default_model=None,
        user_api_key_dict=_actor(models=allowed, project_id="project-a"),
        team=_team(models=allowed, organization_id="org-a"),
        prisma_client=_Client(),
        llm_router=catalog,
        dependency_objects=MemberAutoRouterDependencyObjects(membership, organization, project),
    )
    if restricted is not None:
        with pytest.raises(ProxyException, match="jev-latest"):
            await operation
        return
    await operation
    assert not catalog.get_model_list("typesafe/jev-latest")
