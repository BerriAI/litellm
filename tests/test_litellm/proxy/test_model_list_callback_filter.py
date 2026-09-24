"""
Tests for `CustomLogger.async_filter_listed_models` on the model listing endpoints:
GET /v1/models (`model_list`, OpenAI and Anthropic shapes), GET /v1/models/{id}
(`model_info`), GET /v1/model/info (`model_info_v1`) and GET /model_group/info
(`model_group_info`). A registered callback that overrides the hook decides per
caller which of the names the route would list are kept; the rest disappear and
`/v1/models/{id}` answers 404 for them.
"""

import json
from collections.abc import Sequence

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging


class _Gate(CustomLogger):
    def __init__(self, hidden: frozenset[str] = frozenset(), extra: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.hidden = hidden
        self.extra = extra
        self.seen: list[tuple[str, ...]] = []

    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        self.seen.append(tuple(model_names))
        return [*(name for name in model_names if name not in self.hidden), *self.extra]


class _InferenceOnlyGate(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if data.get("model") == "restricted-model":
            raise HTTPException(status_code=403, detail="not entitled to this model")
        return data


class _RaisingGate(CustomLogger):
    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        raise HTTPException(status_code=503, detail="entitlement service down")


class _ReversingGate(CustomLogger):
    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        return list(reversed(model_names))


class _StringReturningGate(CustomLogger):
    async def async_filter_listed_models(self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]) -> str:
        return "open-model"


def _deployment(model_name: str, model: str = "openai/gpt-4o", **model_info):
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_key": "sk-fake"},
        "model_info": {"id": f"{model_name}-id", **model_info},
    }


def _install_router(monkeypatch, *deployments, **router_kwargs) -> Router:
    router = Router(model_list=list(deployments), **router_kwargs)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    return router


def _register(monkeypatch, *callbacks: CustomLogger) -> None:
    monkeypatch.setattr(litellm, "callbacks", list(callbacks))
    ProxyLogging._callback_capabilities_cache.clear()


@pytest.fixture
def two_model_router(monkeypatch) -> Router:
    return _install_router(monkeypatch, _deployment("open-model"), _deployment("restricted-model"))


@pytest.fixture
def team_router(monkeypatch) -> Router:
    return _install_router(
        monkeypatch,
        _deployment("gpt-4"),
        _deployment("model_name_team1_abc", team_id="team1", team_public_model_name="team-gpt"),
        _deployment("model_name_team1_def", team_id="team1", team_public_model_name="team-chat"),
    )


@pytest.fixture
def team_admin_privileges(monkeypatch) -> None:
    from litellm.proxy.management_endpoints import common_utils

    async def _is_team_admin(**kwargs) -> bool:
        return True

    monkeypatch.setattr(common_utils, "_user_has_admin_privileges", _is_team_admin)


def _non_admin(**kwargs) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.INTERNAL_USER, **kwargs)


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[])


def _team_member() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="u",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="team1",
        team_models=["model_name_team1_abc", "model_name_team1_def"],
        models=["model_name_team1_abc", "model_name_team1_def"],
    )


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


async def _v1_model_info_names(user_api_key_dict: UserAPIKeyAuth) -> list[str]:
    response = await proxy_server.model_info_v1(user_api_key_dict=user_api_key_dict)
    return [row["model_name"] for row in json.loads(response.body)["data"]]


async def _model_groups(user_api_key_dict: UserAPIKeyAuth) -> list[str]:
    response = await proxy_server.model_group_info(user_api_key_dict=user_api_key_dict)
    return [group.model_group for group in response["data"]]


async def _model_by_id_status(model_id: str, user_api_key_dict: UserAPIKeyAuth) -> int:
    try:
        response = await proxy_server.model_info(model_id=model_id, user_api_key_dict=user_api_key_dict)
    except HTTPException as error:
        return error.status_code
    assert response["id"] == model_id
    return 200


@pytest.mark.asyncio
async def test_v1_models_lists_only_the_names_the_callback_keeps(two_model_router, monkeypatch):
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))

    assert await _v1_models(_non_admin()) == ["open-model"]
    assert await _v1_models(_admin()) == ["open-model"]
    assert await _v1_models(_non_admin(), request=_anthropic_request()) == ["open-model"]


@pytest.mark.asyncio
async def test_v1_models_scope_expand_applies_the_callback(two_model_router, team_admin_privileges, monkeypatch):
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))

    assert await _v1_models(_non_admin(), scope="expand") == ["open-model"]
    assert await _v1_models(_admin(), scope="expand") == ["open-model"]


@pytest.mark.asyncio
async def test_v1_models_by_id_answers_404_for_a_name_the_callback_leaves_out(two_model_router, monkeypatch):
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))

    assert await _model_by_id_status("restricted-model", _non_admin()) == 404
    assert await _model_by_id_status("open-model", _non_admin()) == 200


@pytest.mark.asyncio
async def test_v1_model_info_lists_only_the_rows_the_callback_keeps(two_model_router, monkeypatch):
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))

    assert await _v1_model_info_names(_non_admin()) == ["open-model"]
    assert await _v1_model_info_names(_admin()) == ["open-model"]


@pytest.mark.asyncio
async def test_model_group_info_lists_only_the_groups_the_callback_keeps(two_model_router, monkeypatch):
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))

    assert await _model_groups(_non_admin()) == ["open-model"]
    assert await _model_groups(_admin()) == ["open-model"]


@pytest.mark.asyncio
async def test_a_callback_without_the_hook_changes_no_listing(two_model_router, monkeypatch):
    _register(monkeypatch, _InferenceOnlyGate())

    assert await _v1_models(_non_admin()) == ["open-model", "restricted-model"]
    assert await _model_by_id_status("restricted-model", _non_admin()) == 200
    assert await _v1_model_info_names(_non_admin()) == ["open-model", "restricted-model"]
    assert await _model_groups(_non_admin()) == ["open-model", "restricted-model"]


@pytest.mark.asyncio
async def test_callback_cannot_add_a_name_it_was_not_offered(two_model_router, monkeypatch):
    _register(monkeypatch, _Gate(extra=("ghost-model",)))

    assert await _v1_models(_non_admin()) == ["open-model", "restricted-model"]
    assert await _model_by_id_status("ghost-model", _non_admin()) == 404


@pytest.mark.asyncio
async def test_callbacks_narrow_in_registration_order(monkeypatch):
    _install_router(monkeypatch, _deployment("a"), _deployment("b"), _deployment("c"))
    first: _Gate = _Gate(hidden=frozenset({"a"}))
    second: _Gate = _Gate(hidden=frozenset({"b"}))
    _register(monkeypatch, first, second)

    assert await _v1_models(_non_admin()) == ["c"]
    assert first.seen == [("a", "b", "c")]
    assert second.seen == [("b", "c")]


@pytest.mark.asyncio
async def test_callback_sees_and_filters_team_models_by_their_public_name(team_router, monkeypatch):
    gate: _Gate = _Gate(hidden=frozenset({"team-gpt"}))
    _register(monkeypatch, gate)

    assert await _v1_models(_team_member()) == ["team-chat"]
    assert await _model_by_id_status("team-gpt", _team_member()) == 404
    assert await _model_by_id_status("team-chat", _team_member()) == 200
    assert all("team-gpt" in seen and "model_name_team1_abc" not in seen for seen in gate.seen)


@pytest.mark.asyncio
async def test_callback_sees_public_team_names_on_every_listing_route(team_router, monkeypatch):
    gate: _Gate = _Gate(hidden=frozenset({"team-gpt"}))
    _register(monkeypatch, gate)

    assert await _v1_models(_team_member()) == ["team-chat"]
    assert await _v1_model_info_names(_team_member()) == ["team-chat"]
    assert await _model_groups(_team_member()) == ["model_name_team1_def"]
    assert await _model_by_id_status("team-gpt", _team_member()) == 404
    assert len(gate.seen) == 4
    assert all(sorted(seen) == ["team-chat", "team-gpt"] for seen in gate.seen)


@pytest.mark.asyncio
async def test_router_alias_follows_its_hidden_target(monkeypatch):
    _install_router(
        monkeypatch,
        _deployment("open-model"),
        _deployment("restricted-model"),
        model_group_alias={"mini": "restricted-model", "wide": "open-model"},
    )
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))

    assert sorted(await _v1_models(_non_admin())) == ["open-model", "wide"]
    assert sorted(await _v1_model_info_names(_non_admin())) == ["open-model", "wide"]
    assert sorted(await _model_groups(_non_admin())) == ["open-model", "wide"]

    _register(monkeypatch, _Gate(hidden=frozenset({"mini"})))

    assert sorted(await _v1_models(_non_admin())) == ["open-model", "restricted-model", "wide"]


@pytest.mark.asyncio
async def test_v1_model_info_offers_only_the_rows_the_caller_would_see(monkeypatch):
    _install_router(monkeypatch, _deployment("open-model"), _deployment("hidden-model", discoverable=False))
    gate: _Gate = _Gate()
    _register(monkeypatch, gate)

    assert await _v1_model_info_names(_non_admin()) == ["open-model"]
    assert await _v1_model_info_names(_admin()) == ["open-model", "hidden-model"]
    assert gate.seen == [("open-model",), ("open-model", "hidden-model")]


@pytest.mark.asyncio
async def test_listing_keeps_its_order_whatever_order_the_callback_returns(monkeypatch):
    _install_router(monkeypatch, _deployment("a"), _deployment("b"), _deployment("c"))
    _register(monkeypatch, _ReversingGate())

    assert await _v1_models(_non_admin()) == ["a", "b", "c"]
    assert await _v1_model_info_names(_non_admin()) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_callback_returning_a_string_is_an_error_not_an_empty_listing(two_model_router, monkeypatch):
    _register(monkeypatch, _StringReturningGate())

    with pytest.raises(TypeError, match=r"_StringReturningGate\.async_filter_listed_models"):
        await _v1_models(_non_admin())


@pytest.mark.asyncio
async def test_alias_of_a_hidden_model_is_not_listed(two_model_router, monkeypatch):
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))
    caller = _non_admin(aliases={"mini": "restricted-model", "wide": "open-model"})

    assert await _v1_models(caller) == ["open-model", "wide"]
    assert await _model_by_id_status("mini", caller) == 404
    assert await _model_by_id_status("wide", caller) == 200


@pytest.mark.asyncio
async def test_callback_error_reaches_the_caller(two_model_router, monkeypatch):
    _register(monkeypatch, _RaisingGate())

    with pytest.raises(HTTPException) as raised:
        await _v1_models(_non_admin())
    assert raised.value.status_code == 503


@pytest.mark.asyncio
async def test_hidden_model_still_routes_for_direct_requests(two_model_router, monkeypatch):
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))
    assert "restricted-model" not in await _v1_models(_non_admin())

    deployment = two_model_router.get_available_deployment(
        model="restricted-model", messages=[{"role": "user", "content": "hi"}]
    )
    assert deployment["model_name"] == "restricted-model"


@pytest.mark.asyncio
async def test_model_group_info_offers_a2a_agent_groups_to_the_callback(two_model_router, monkeypatch):
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.types.agents import AgentResponse

    monkeypatch.setattr(
        global_agent_registry,
        "agent_list",
        [AgentResponse(agent_id="agent-1", agent_name="helper", agent_card_params={})],
    )
    caller = _non_admin(object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="p1", agents=["agent-1"]))
    gate: _Gate = _Gate()
    _register(monkeypatch, gate)

    assert await _model_groups(caller) == ["open-model", "restricted-model", "a2a/helper"]
    assert gate.seen == [("open-model", "restricted-model", "a2a/helper")]

    _register(monkeypatch, _Gate(hidden=frozenset({"a2a/helper", "restricted-model"})))

    assert await _model_groups(caller) == ["open-model"]


async def _v1_model_info_by_deployment_id(deployment_id: str, user_api_key_dict: UserAPIKeyAuth) -> int | list[str]:
    try:
        response = await proxy_server.model_info_v1(user_api_key_dict=user_api_key_dict, litellm_model_id=deployment_id)
    except HTTPException as error:
        return error.status_code
    return [row["model_name"] for row in json.loads(response.body)["data"]]


@pytest.mark.asyncio
async def test_v1_model_info_by_deployment_id_answers_like_an_unknown_id_for_a_hidden_model(
    two_model_router, monkeypatch
):
    _register(monkeypatch, _Gate(hidden=frozenset({"restricted-model"})))

    assert await _v1_model_info_by_deployment_id("restricted-model-id", _non_admin()) == 400
    assert await _v1_model_info_by_deployment_id("no-such-id", _non_admin()) == 400
    assert await _v1_model_info_by_deployment_id("open-model-id", _non_admin()) == ["open-model"]


@pytest.mark.asyncio
async def test_v1_model_info_by_deployment_id_offers_the_public_team_name(team_router, monkeypatch):
    gate: _Gate = _Gate(hidden=frozenset({"team-gpt"}))
    _register(monkeypatch, gate)

    assert await _v1_model_info_by_deployment_id("model_name_team1_abc-id", _team_member()) == 400
    assert await _v1_model_info_by_deployment_id("model_name_team1_def-id", _team_member()) == ["team-chat"]
    assert gate.seen == [("team-gpt",), ("team-chat",)]
