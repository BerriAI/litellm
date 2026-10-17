from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LiteLLM_VerificationToken,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_endpoints.key_management_endpoints import info_key_fn
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

HASH: Final = "a" * 64


@dataclass(frozen=True)
class KeyInfoContext:
    cache: UserApiKeyCache
    database: MagicMock
    router: Router
    key: LiteLLM_VerificationToken


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> KeyInfoContext:
    router: Final = Router(
        model_list=[
            {
                "model_name": "model-a",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"},
                "model_info": {"access_groups": ["big", "small"]},
            },
            {
                "model_name": "model-b",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"},
                "model_info": {"access_groups": ["big"]},
            },
            {
                "model_name": "other-team-model",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"},
                "model_info": {"access_groups": ["big"], "team_id": "other-team"},
            },
        ]
    )
    cache: Final = UserApiKeyCache()
    cache.set_cache("owner", LiteLLM_UserTable(user_id="owner", models=[]))
    cache.set_cache("team_id:team", LiteLLM_TeamTableCachedObj(team_id="team", models=["small"]))
    key: Final = LiteLLM_VerificationToken(token=HASH, user_id="owner", models=["big", "small"])
    database: Final = MagicMock()
    database.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=key)
    database.db.litellm_teammembership.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    return KeyInfoContext(cache, database, router, key)


async def test_hash_lookup_expands_groups_without_changing_stored_grants(context: KeyInfoContext) -> None:
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result["info"]["models"] == ["big", "small"]
    assert result["info"]["effective_models"] == ["model-a", "model-b"]
    assert context.key.models == ["big", "small"]
    context.database.db.litellm_verificationtoken.find_unique.assert_awaited_once_with(
        where={"token": HASH},
        include={"litellm_budget_table": True},
    )


async def test_omitting_flag_preserves_existing_response(context: KeyInfoContext) -> None:
    result: Final = await info_key_fn(
        key=HASH, user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    )
    assert "effective_models" not in result["info"]
    assert result["info"]["models"] == ["big", "small"]


async def test_admin_lookup_uses_target_team_ceiling_not_admin_access(context: KeyInfoContext) -> None:
    context.database.db.litellm_verificationtoken.find_unique.return_value = context.key.model_copy(
        update={"team_id": "team"}
    )
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, models=["all-proxy-models"]),
    )
    assert result["info"]["models"] == ["big", "small"]
    assert result["info"]["effective_models"] == ["model-a"]


@pytest.mark.parametrize("grants", [[], ["all-proxy-models"], ["model-*"], ["model-a", "big"]])
async def test_legacy_grants_list_only_eligible_registered_models(context: KeyInfoContext, grants: list[str]) -> None:
    context.database.db.litellm_verificationtoken.find_unique.return_value = context.key.model_copy(
        update={"models": grants}
    )
    result: Final = await info_key_fn(
        key=HASH, include_effective_models=True, user_api_key_dict=UserAPIKeyAuth(user_id="owner")
    )
    assert result["info"]["models"] == grants
    assert result["info"]["effective_models"] == ["model-a", "model-b"]


async def test_another_user_cannot_expand_target_key(context: KeyInfoContext) -> None:
    with pytest.raises(ProxyException) as error:
        await info_key_fn(
            key=HASH,
            include_effective_models=True,
            user_api_key_dict=UserAPIKeyAuth(user_id="stranger", user_role=LitellmUserRoles.INTERNAL_USER),
        )
    assert str(error.value.code) == "403"


@pytest.mark.parametrize("state", ["blocked", "expired", "deleted"])
async def test_inactive_key_has_no_effective_models(context: KeyInfoContext, state: str) -> None:
    row: Final = context.key.model_copy(
        update={
            "blocked": state == "blocked",
            "expires": datetime.now(timezone.utc) - timedelta(days=1) if state == "expired" else None,
        }
    )
    if state == "deleted":
        context.database.db.litellm_verificationtoken.find_unique.return_value = None
        context.database.db.litellm_deletedverificationtoken.find_first = AsyncMock(return_value=row)
    else:
        context.database.db.litellm_verificationtoken.find_unique.return_value = row
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result["info"]["effective_models"] == []


async def test_group_change_is_visible_without_rewriting_key(context: KeyInfoContext) -> None:
    caller: Final = UserAPIKeyAuth(user_id="owner")
    before: Final = await info_key_fn(key=HASH, include_effective_models=True, user_api_key_dict=caller)
    context.router.add_deployment(
        Deployment(
            model_name="model-c",
            litellm_params=LiteLLM_Params(model="openai/gpt-4o-mini", api_key="fake"),
            model_info=ModelInfo(access_groups=["big"]),
        )
    )
    after: Final = await info_key_fn(key=HASH, include_effective_models=True, user_api_key_dict=caller)
    assert before["info"]["effective_models"] == ["model-a", "model-b"]
    assert after["info"]["effective_models"] == ["model-a", "model-b", "model-c"]
    assert before["info"]["models"] == after["info"]["models"] == ["big", "small"]


async def test_personal_key_obeys_owner_models(context: KeyInfoContext) -> None:
    context.cache.set_cache("owner", LiteLLM_UserTable(user_id="owner", models=["small"]))
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result["info"]["effective_models"] == ["model-a"]


async def test_blocked_team_has_no_effective_models(context: KeyInfoContext) -> None:
    context.cache.set_cache("team_id:team", LiteLLM_TeamTableCachedObj(team_id="team", models=["big"], blocked=True))
    context.database.db.litellm_verificationtoken.find_unique.return_value = context.key.model_copy(
        update={"team_id": "team"}
    )
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result["info"]["effective_models"] == []


async def test_http_query_flag_adds_effective_models(context: KeyInfoContext) -> None:
    import httpx
    from fastapi import FastAPI

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.key_management_endpoints import router

    app: Final = FastAPI()
    app.include_router(router)

    async def caller() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    app.dependency_overrides[user_api_key_auth] = caller
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response: Final = await client.get(f"/key/info?key={HASH}&include_effective_models=true")
    assert response.status_code == 200, response.text
    assert response.json()["info"]["models"] == ["big", "small"]
    assert response.json()["info"]["effective_models"] == ["model-a", "model-b"]


async def test_key_group_is_a_ceiling_even_when_other_routes_exist(context: KeyInfoContext) -> None:
    context.database.db.litellm_verificationtoken.find_unique.return_value = context.key.model_copy(
        update={"models": ["small"]}
    )
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result["info"]["effective_models"] == ["model-a"]


async def test_project_models_narrow_key_groups(context: KeyInfoContext) -> None:
    from litellm.proxy._types import LiteLLM_ProjectTableCachedObj
    from litellm.proxy.common_utils.user_api_key_cache import project_cache_key

    context.cache.set_cache(
        project_cache_key("project"), LiteLLM_ProjectTableCachedObj(project_id="project", models=["small"])
    )
    context.database.db.litellm_verificationtoken.find_unique.return_value = context.key.model_copy(
        update={"project_id": "project"}
    )
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result["info"]["effective_models"] == ["model-a"]


async def test_wildcard_routes_are_not_enumerated(context: KeyInfoContext) -> None:
    context.router.add_deployment(
        Deployment(model_name="openai/*", litellm_params=LiteLLM_Params(model="openai/*", api_key="fake"))
    )
    context.database.db.litellm_verificationtoken.find_unique.return_value = context.key.model_copy(
        update={"models": ["all-proxy-models"]}
    )
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result["info"]["effective_models"] == ["model-a", "model-b"]


@pytest.mark.parametrize("public_name", [None, "project-model"])
async def test_matching_team_can_list_its_routes(context: KeyInfoContext, public_name: str | None) -> None:
    route_name: Final = "model_name_team_example" if public_name else "team-route"
    context.router.add_deployment(
        Deployment(
            model_name=route_name,
            litellm_params=LiteLLM_Params(model="openai/gpt-4o-mini", api_key="fake"),
            model_info=ModelInfo(team_id="team", team_public_model_name=public_name, access_groups=["big"]),
        )
    )
    context.cache.set_cache("team_id:team", LiteLLM_TeamTableCachedObj(team_id="team", models=["big"]))
    context.database.db.litellm_verificationtoken.find_unique.return_value = context.key.model_copy(
        update={"team_id": "team"}
    )
    result: Final = await info_key_fn(
        key=HASH,
        include_effective_models=True,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result["info"]["effective_models"] == sorted(["model-a", "model-b", public_name or route_name])
