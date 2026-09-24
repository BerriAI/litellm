"""
Tests for key and team `model_aliases` on the model listing endpoints: GET /v1/models
(`model_list`, OpenAI and Anthropic shapes) and GET /v1/models/{id} (`model_info`).
An alias the caller can complete on is listed next to its target and resolves by name.
"""

import pytest
from starlette.requests import Request

from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _deployment(model_name: str, model: str = "openai/gpt-4.1-mini", **model_info):
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_key": "sk-fake"},
        "model_info": {"id": f"{model_name}-id", **model_info},
    }


@pytest.fixture
def router(monkeypatch) -> Router:
    router = Router(
        model_list=[
            _deployment("gpt-4.1-mini"),
            _deployment("gpt-4.1", model="openai/gpt-4.1"),
            _deployment("model_name_team1_abc", team_id="team1", team_public_model_name="team-chat"),
            _deployment("hidden", model="anthropic/claude-sonnet-4-5", discoverable=False),
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    return router


def _team_member(
    team_id: str = "team1", models: list[str] | None = None, **aliases: dict[str, str] | None
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="u",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id=team_id,
        team_models=["gpt-4.1-mini", "model_name_team1_abc"],
        models=models or ["gpt-4.1-mini", "model_name_team1_abc"],
        **aliases,
    )


def _anthropic_request(*extra_headers: tuple[bytes, bytes]) -> Request:
    return Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "query_string": b"",
            "headers": [(b"anthropic-version", b"2023-06-01"), *extra_headers],
        }
    )


def _claude_code_request() -> Request:
    return _anthropic_request((b"user-agent", b"claude-cli/2.1.267 (external, cli)"))


async def _v1_models(user_api_key_dict: UserAPIKeyAuth, **kwargs) -> list[str]:
    response = await proxy_server.model_list(user_api_key_dict=user_api_key_dict, **kwargs)
    return [m["id"] for m in response["data"]]


@pytest.mark.asyncio
async def test_v1_models_lists_team_alias_next_to_its_target_in_both_shapes(router):
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": "gpt-4.1-mini"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "claude-sonnet-4-5"]
    assert await _v1_models(caller, request=_anthropic_request()) == ["gpt-4.1-mini", "team-chat", "claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_claude_code_picker_lists_the_alias_under_its_own_name(router):
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": "gpt-4.1-mini"})

    picker_ids = await _v1_models(caller, request=_claude_code_request())
    assert any(picker_id.startswith("claude-sonnet-4-5") for picker_id in picker_ids), picker_ids


@pytest.mark.asyncio
async def test_v1_models_lists_key_alias_and_hides_alias_to_a_model_the_caller_cannot_list(router):
    caller = _team_member(aliases={"mini": "gpt-4.1-mini", "big": "gpt-4.1"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "mini"]


@pytest.mark.asyncio
async def test_v1_models_resolves_a_team_alias_through_the_key_alias_like_chat_completions_does(router):
    caller = _team_member(team_model_aliases={"fast": "mid"}, aliases={"fast": "gpt-4.1", "mid": "gpt-4.1-mini"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "fast", "mid"]
    response = await proxy_server.model_info(model_id="fast", user_api_key_dict=caller)
    assert response["id"] == "fast"


@pytest.mark.asyncio
async def test_v1_models_skips_only_the_malformed_alias_entries(router):
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": 5, "fast": "gpt-4.1-mini"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "fast"]


@pytest.mark.asyncio
async def test_v1_models_by_id_resolves_a_team_alias_to_its_target_metadata(router):
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": "gpt-4.1-mini"})

    response = await proxy_server.model_info(model_id="claude-sonnet-4-5", user_api_key_dict=caller)
    assert response["id"] == "claude-sonnet-4-5"
    assert response["owned_by"] == "openai"


@pytest.mark.asyncio
async def test_v1_models_by_id_retrieves_the_listed_model_when_an_alias_collides_with_its_id(router):
    caller = _team_member(aliases={"team-chat": "gpt-4.1"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat"]
    response = await proxy_server.model_info(model_id="team-chat", user_api_key_dict=caller)
    assert response["id"] == "team-chat"


@pytest.mark.asyncio
async def test_v1_models_by_id_resolves_an_alias_named_like_an_undiscoverable_model_to_the_alias_target(router):
    caller = _team_member(aliases={"hidden": "gpt-4.1-mini"}, models=["gpt-4.1-mini", "model_name_team1_abc", "hidden"])

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "hidden"]
    target = await proxy_server.model_info(model_id="gpt-4.1-mini", user_api_key_dict=caller)
    response = await proxy_server.model_info(model_id="hidden", user_api_key_dict=caller)
    assert response == {**target, "id": "hidden"}


@pytest.mark.asyncio
async def test_v1_models_by_id_keeps_the_alias_as_id_when_it_targets_a_team_scoped_model(router):
    caller = _team_member(team_model_aliases={"chat": "team-chat"})

    assert "chat" in await _v1_models(caller)
    response = await proxy_server.model_info(model_id="chat", user_api_key_dict=caller)
    assert response["id"] == "chat"
