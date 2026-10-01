"""
Tests for `model_info.discoverable: false` on the model listing endpoints:
GET /v1/models (`model_list`, OpenAI and Anthropic shapes), GET /v1/models/{id}
(`model_info`), GET /v1/model/info (`model_info_v1`) and GET /model_group/info
(`model_group_info`). Flagged models drop out of the listings for callers without
the admin view and stay reachable by name.
"""

import json

import pytest
from starlette.requests import Request

from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _deployment(model_name: str, model: str = "openai/gpt-4o", **model_info):
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_key": "sk-fake"},
        "model_info": {"id": f"{model_name}-id", **model_info},
    }


def _install_router(monkeypatch, *deployments) -> Router:
    router = Router(model_list=list(deployments))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    return router


@pytest.fixture
def flagged_router(monkeypatch) -> Router:
    return _install_router(
        monkeypatch,
        _deployment("gpt-4"),
        _deployment("internal-evaluator", discoverable=False),
    )


@pytest.fixture
def flagged_wildcard_router(monkeypatch) -> Router:
    return _install_router(
        monkeypatch,
        _deployment("gpt-4"),
        _deployment("anthropic/*", model="anthropic/*", discoverable=False),
    )


@pytest.fixture
def flagged_team_router(monkeypatch) -> Router:
    return _install_router(
        monkeypatch,
        _deployment("gpt-4"),
        _deployment(
            "model_name_team1_abc", team_id="team1", team_public_model_name="team-gpt", discoverable=False
        ),
        _deployment("model_name_team1_def", team_id="team1", team_public_model_name="team-chat"),
    )


@pytest.fixture
def team_admin_privileges(monkeypatch) -> None:
    from litellm.proxy.management_endpoints import common_utils

    async def _is_team_admin(**kwargs) -> bool:
        return True

    monkeypatch.setattr(common_utils, "_user_has_admin_privileges", _is_team_admin)


def _non_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.INTERNAL_USER)


def _team_member(role: LitellmUserRoles = LitellmUserRoles.INTERNAL_USER) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test", user_id="u", user_role=role, team_id="team1", team_models=["team-gpt", "team-chat"]
    )


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[])


def _anthropic_request() -> Request:
    return Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "query_string": b"",
            "headers": [(b"anthropic-version", b"2023-06-01")],
        }
    )


async def _v1_models(user_api_key_dict: UserAPIKeyAuth, **kwargs) -> list[str]:
    response = await proxy_server.model_list(user_api_key_dict=user_api_key_dict, **kwargs)
    return [m["id"] for m in response["data"]]


async def _v1_model_info_names(user_api_key_dict: UserAPIKeyAuth, **kwargs) -> list[str]:
    response = await proxy_server.model_info_v1(user_api_key_dict=user_api_key_dict, **kwargs)
    return [row["model_name"] for row in json.loads(response.body)["data"]]


async def _model_groups(user_api_key_dict: UserAPIKeyAuth) -> list[str]:
    response = await proxy_server.model_group_info(user_api_key_dict=user_api_key_dict)
    return [group.model_group for group in response["data"]]


@pytest.mark.asyncio
async def test_v1_models_openai_shape_hides_flagged_model_from_non_admin_only(flagged_router):
    assert await _v1_models(_non_admin()) == ["gpt-4"]
    assert await _v1_models(_admin()) == ["gpt-4", "internal-evaluator"]


@pytest.mark.asyncio
async def test_v1_models_anthropic_shape_hides_flagged_model_from_non_admin_only(flagged_router):
    assert await _v1_models(_non_admin(), request=_anthropic_request()) == ["gpt-4"]
    assert await _v1_models(_admin(), request=_anthropic_request()) == ["gpt-4", "internal-evaluator"]


@pytest.mark.asyncio
async def test_v1_models_scope_expand_hides_flagged_model_from_team_admin_only(flagged_router, team_admin_privileges):
    assert await _v1_models(_non_admin(), scope="expand") == ["gpt-4"]
    assert await _v1_models(_admin(), scope="expand") == ["gpt-4", "internal-evaluator"]


@pytest.mark.asyncio
async def test_v1_models_by_id_still_serves_the_hidden_model_to_non_admin(flagged_router):
    assert "internal-evaluator" not in await _v1_models(_non_admin())

    response = await proxy_server.model_info(model_id="internal-evaluator", user_api_key_dict=_non_admin())
    assert response["id"] == "internal-evaluator"


@pytest.mark.asyncio
async def test_v1_models_group_with_one_discoverable_deployment_stays_listed(monkeypatch):
    _install_router(
        monkeypatch,
        _deployment("shared", discoverable=False),
        {
            "model_name": "shared",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"},
            "model_info": {"id": "shared-public"},
        },
        _deployment("internal-evaluator", discoverable=False),
    )

    assert await _v1_models(_non_admin()) == ["shared"]


@pytest.mark.asyncio
async def test_v1_models_only_an_explicit_false_hides_a_model(monkeypatch):
    _install_router(
        monkeypatch,
        _deployment("gpt-4"),
        _deployment("public-eval", discoverable=True),
        _deployment("internal-evaluator", discoverable=False),
    )

    assert await _v1_models(_non_admin()) == ["gpt-4", "public-eval"]


@pytest.mark.asyncio
async def test_v1_model_info_hides_flagged_rows_from_non_admin_only(flagged_router):
    assert await _v1_model_info_names(_non_admin()) == ["gpt-4"]
    assert await _v1_model_info_names(_admin()) == ["gpt-4", "internal-evaluator"]


@pytest.mark.asyncio
async def test_v1_model_info_by_id_still_serves_the_hidden_row_to_non_admin(flagged_router):
    assert "internal-evaluator" not in await _v1_model_info_names(_non_admin())

    assert await _v1_model_info_names(_non_admin(), litellm_model_id="internal-evaluator-id") == [
        "internal-evaluator"
    ]


@pytest.mark.asyncio
async def test_model_group_info_hides_flagged_group_from_non_admin_only(flagged_router):
    assert await _model_groups(_non_admin()) == ["gpt-4"]
    assert await _model_groups(_admin()) == ["gpt-4", "internal-evaluator"]


@pytest.mark.asyncio
async def test_v1_models_hides_flagged_team_model_from_its_team_member_only(flagged_team_router):
    assert await _v1_models(_team_member()) == ["team-chat"]
    assert set(await _v1_models(_team_member(LitellmUserRoles.PROXY_ADMIN))) >= {"team-gpt", "team-chat"}


@pytest.mark.asyncio
async def test_model_group_info_hides_flagged_team_model_from_its_team_member(flagged_team_router):
    assert await _model_groups(_team_member()) == ["team-chat"]


@pytest.mark.asyncio
async def test_v1_models_hides_flagged_wildcard_expansions_from_non_admin(flagged_wildcard_router):
    assert await _v1_models(_non_admin(), return_wildcard_routes=True) == ["gpt-4"]

    admin_ids = await _v1_models(_admin(), return_wildcard_routes=True)
    assert "gpt-4" in admin_ids
    assert any(model_id.startswith("anthropic/") for model_id in admin_ids)


@pytest.mark.asyncio
async def test_v1_model_info_hides_flagged_wildcard_expanded_rows_from_non_admin(flagged_wildcard_router):
    assert await _v1_model_info_names(_non_admin()) == ["gpt-4"]

    admin_names = await _v1_model_info_names(_admin())
    assert "gpt-4" in admin_names
    assert any(name.startswith("anthropic/") for name in admin_names)


@pytest.mark.asyncio
async def test_hidden_model_still_routes_for_direct_requests(flagged_router):
    assert "internal-evaluator" not in await _v1_models(_non_admin())

    deployment = flagged_router.get_available_deployment(
        model="internal-evaluator", messages=[{"role": "user", "content": "hi"}]
    )
    assert deployment["model_name"] == "internal-evaluator"
