"""
Tests for key and team `model_aliases` on the model listing endpoints: GET /v1/models
(`model_list`, OpenAI and Anthropic shapes) and GET /v1/models/{id} (`model_info`).
An alias the caller can complete on is listed next to its target and resolves by name.
"""

import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _deployment(model_name: str, model: str = "openai/gpt-4.1-mini", **model_info: str | bool) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_key": "sk-fake"},
        "model_info": {"id": f"{model_name}-id", **model_info},
    }


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> Router:
    router = Router(
        model_list=[
            _deployment("gpt-4.1-mini"),
            _deployment("gpt-4.1", model="openai/gpt-4.1", display_name="GPT 4.1"),
            _deployment(
                "model_name_team1_abc", model="openai/gpt-4o", team_id="team1", team_public_model_name="team-chat"
            ),
            _deployment("hidden", model="anthropic/claude-sonnet-4-5", discoverable=False),
            _deployment("paused", model="openai/gpt-4.1", blocked=True),
        ],
        model_group_alias={"legacy": "gpt-4.1"},
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


async def _v1_models(user_api_key_dict: UserAPIKeyAuth, request: Request | None = None) -> list[str]:
    response = await proxy_server.model_list(user_api_key_dict=user_api_key_dict, request=request)
    return [m["id"] for m in response["data"]]


@pytest.mark.asyncio
async def test_v1_models_lists_team_alias_next_to_its_target_in_both_shapes(router: Router) -> None:
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": "gpt-4.1-mini"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "claude-sonnet-4-5"]
    assert await _v1_models(caller, request=_anthropic_request()) == ["gpt-4.1-mini", "team-chat", "claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_claude_code_picker_lists_the_alias_under_its_own_name(router: Router) -> None:
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": "gpt-4.1-mini"})

    picker_ids = await _v1_models(caller, request=_claude_code_request())
    assert any(picker_id.startswith("claude-sonnet-4-5") for picker_id in picker_ids), picker_ids


@pytest.mark.asyncio
async def test_v1_models_lists_key_alias_and_hides_alias_to_a_model_the_caller_cannot_list(router: Router) -> None:
    caller = _team_member(aliases={"mini": "gpt-4.1-mini", "big": "gpt-4.1"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "mini"]


@pytest.mark.asyncio
async def test_v1_models_resolves_a_team_alias_through_the_key_alias_like_chat_completions_does(router: Router) -> None:
    caller = _team_member(team_model_aliases={"fast": "mid"}, aliases={"fast": "gpt-4.1", "mid": "gpt-4.1-mini"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "fast", "mid"]
    response = await proxy_server.model_info(model_id="fast", user_api_key_dict=caller)
    assert response["id"] == "fast"


@pytest.mark.asyncio
async def test_v1_models_skips_only_the_malformed_alias_entries(router: Router) -> None:
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": 5, "fast": "gpt-4.1-mini"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "fast"]


@pytest.mark.asyncio
async def test_v1_models_by_id_resolves_a_team_alias_to_its_target_metadata(router: Router) -> None:
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": "gpt-4.1-mini"})

    response = await proxy_server.model_info(model_id="claude-sonnet-4-5", user_api_key_dict=caller)
    assert response["id"] == "claude-sonnet-4-5"
    assert response["owned_by"] == "openai"


@pytest.mark.asyncio
async def test_v1_models_by_id_retrieves_the_listed_model_when_an_alias_collides_with_its_id(router: Router) -> None:
    caller = _team_member(aliases={"team-chat": "gpt-4.1"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat"]
    response = await proxy_server.model_info(model_id="team-chat", user_api_key_dict=caller)
    assert response["id"] == "team-chat"


@pytest.mark.asyncio
async def test_v1_models_by_id_resolves_an_alias_named_like_an_undiscoverable_model_to_the_alias_target(
    router: Router,
) -> None:
    caller = _team_member(aliases={"hidden": "gpt-4.1-mini"}, models=["gpt-4.1-mini", "model_name_team1_abc", "hidden"])

    assert await _v1_models(caller) == ["gpt-4.1-mini", "team-chat", "hidden"]
    target = await proxy_server.model_info(model_id="gpt-4.1-mini", user_api_key_dict=caller)
    response = await proxy_server.model_info(model_id="hidden", user_api_key_dict=caller)
    assert response == {**target, "id": "hidden"}


@pytest.mark.asyncio
async def test_v1_models_by_id_keeps_the_alias_as_id_when_it_targets_a_team_scoped_model(router: Router) -> None:
    caller = _team_member(team_model_aliases={"chat": "team-chat"})

    assert "chat" in await _v1_models(caller)
    response = await proxy_server.model_info(model_id="chat", user_api_key_dict=caller)
    assert response["id"] == "chat"


@pytest.mark.asyncio
async def test_alias_the_key_also_lists_reports_its_target_metadata_at_list_and_retrieval(router: Router) -> None:
    caller = _team_member(
        models=["gpt-4.1-mini", "model_name_team1_abc", "fast"], team_model_aliases={"fast": "gpt-4.1-mini"}
    )

    response = await proxy_server.model_list(user_api_key_dict=caller, request=None)
    rows = {row["id"]: row for row in response["data"]}
    assert rows["fast"] == {**rows["gpt-4.1-mini"], "id": "fast"}
    target = await proxy_server.model_info(model_id="gpt-4.1-mini", user_api_key_dict=caller)
    assert await proxy_server.model_info(model_id="fast", user_api_key_dict=caller) == {**target, "id": "fast"}


def _keyholder(*models: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="u", user_role=LitellmUserRoles.INTERNAL_USER, models=list(models))


@pytest.mark.asyncio
async def test_team_alias_to_a_model_the_team_does_not_list_reports_that_model_at_list_and_retrieval(
    router: Router,
) -> None:
    caller = _team_member(models=["gpt-4.1-mini", "model_name_team1_abc", "fast"], team_model_aliases={"fast": "gpt-4.1"})
    target_holder = _keyholder("gpt-4.1")

    rows = {row["id"]: row for row in (await proxy_server.model_list(user_api_key_dict=caller, request=None))["data"]}
    target_rows = (await proxy_server.model_list(user_api_key_dict=target_holder, request=None))["data"]
    assert [rows["fast"]] == [{**row, "id": "fast"} for row in target_rows if row["id"] == "gpt-4.1"]
    target = await proxy_server.model_info(model_id="gpt-4.1", user_api_key_dict=target_holder)
    assert await proxy_server.model_info(model_id="fast", user_api_key_dict=caller) == {**target, "id": "fast"}


@pytest.mark.asyncio
async def test_alias_named_like_a_listed_model_keeps_that_model_s_metadata_at_list_and_retrieval(
    router: Router,
) -> None:
    plain = await proxy_server.model_list(user_api_key_dict=_team_member(), request=None)
    shadowing = _team_member(team_model_aliases={"gpt-4.1-mini": "team-chat"})

    assert (await proxy_server.model_list(user_api_key_dict=shadowing, request=None))["data"] == plain["data"]
    assert await proxy_server.model_info(model_id="gpt-4.1-mini", user_api_key_dict=shadowing) == (
        await proxy_server.model_info(model_id="gpt-4.1-mini", user_api_key_dict=_team_member())
    )


@pytest.mark.asyncio
async def test_router_alias_the_caller_lists_resolves_at_retrieval_like_its_listing_row(router: Router) -> None:
    caller = _keyholder("gpt-4.1", "legacy")

    rows = {row["id"]: row for row in (await proxy_server.model_list(user_api_key_dict=caller, request=None))["data"]}
    assert rows["legacy"] == {**rows["gpt-4.1"], "id": "legacy"}
    target = await proxy_server.model_info(model_id="gpt-4.1", user_api_key_dict=caller)
    assert await proxy_server.model_info(model_id="legacy", user_api_key_dict=caller) == {**target, "id": "legacy"}


def _alias_holder(**aliases: dict[str, object]) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="u",
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=["gpt-4.1", "fast"],
        **aliases,
    )


@pytest.mark.asyncio
async def test_listed_alias_keeps_its_own_display_name_in_the_anthropic_and_codex_listings(router: Router) -> None:
    caller = _alias_holder(aliases={"fast": "gpt-4.1"})

    anthropic = await proxy_server.model_list(user_api_key_dict=caller, request=_anthropic_request())
    assert {row["id"]: row["display_name"] for row in anthropic["data"]} == {"gpt-4.1": "GPT 4.1", "fast": "fast"}
    codex = await proxy_server.model_list(user_api_key_dict=caller, request=None, client_version="0.200.0")
    target, alias = json.loads(codex.body)["models"]
    assert (target["display_name"], alias["display_name"]) == ("GPT 4.1", "fast")
    assert {**alias, "slug": "gpt-4.1", "display_name": "GPT 4.1", "priority": 0} == target


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_group_alias", [{"fast": "gpt-4.1"}, {"fast": {"model": "gpt-4.1", "hidden": True}}]
)
async def test_key_router_settings_alias_the_key_lists_reports_its_target_at_list_and_retrieval(
    router: Router, model_group_alias: dict[str, object]
) -> None:
    caller = _alias_holder(router_settings={"model_group_alias": model_group_alias})

    rows = {row["id"]: row for row in (await proxy_server.model_list(user_api_key_dict=caller, request=None))["data"]}
    assert rows["fast"] == {**rows["gpt-4.1"], "id": "fast"}
    target = await proxy_server.model_info(model_id="gpt-4.1", user_api_key_dict=caller)
    assert await proxy_server.model_info(model_id="fast", user_api_key_dict=caller) == {**target, "id": "fast"}


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["paused", "hidden"])
async def test_alias_to_a_model_the_listing_hides_is_not_described_by_it(router: Router, target: str) -> None:
    models = ["gpt-4.1-mini", "model_name_team1_abc", "fast"]
    caller = _team_member(models=models, team_model_aliases={"fast": target})

    listing = await proxy_server.model_list(user_api_key_dict=caller, request=None, include_metadata=True)
    unaliased = await proxy_server.model_list(
        user_api_key_dict=_team_member(models=models), request=None, include_metadata=True
    )
    assert listing["data"] == unaliased["data"]
    with pytest.raises(HTTPException) as retrieval:
        await proxy_server.model_info(model_id="fast", user_api_key_dict=caller)
    assert retrieval.value.status_code == 404
