from unittest.mock import AsyncMock

import pytest

from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
from litellm.proxy._types import LiteLLM_SkillsTable, NewSkillRequest, UserAPIKeyAuth


def _skill(skill_id: str, created_by: str | None) -> LiteLLM_SkillsTable:
    return LiteLLM_SkillsTable(
        skill_id=skill_id,
        display_title="skill",
        created_by=created_by,
    )


@pytest.mark.asyncio
async def test_should_store_team_owner_for_keys_without_user_id(monkeypatch):
    table = AsyncMock()
    table.create.side_effect = lambda data: _skill(data["skill_id"], data["created_by"])
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    auth = UserAPIKeyAuth(team_id="team-1")

    skill = await LiteLLMSkillsHandler.create_skill(
        data=NewSkillRequest(display_title="skill"),
        user_api_key_dict=auth,
    )

    assert skill.created_by == "team:team-1"
    assert table.create.await_args.kwargs["data"]["updated_by"] == "team:team-1"


@pytest.mark.asyncio
async def test_should_filter_list_skills_to_authenticated_owner_scopes(monkeypatch):
    table = AsyncMock()
    table.find_many.return_value = [_skill("litellm_skill_owner", "user-1")]
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    auth = UserAPIKeyAuth(user_id="user-1", team_id="team-1")

    skills = await LiteLLMSkillsHandler.list_skills(user_api_key_dict=auth)

    assert [skill.skill_id for skill in skills] == ["litellm_skill_owner"]
    table.find_many.assert_awaited_once()
    where = table.find_many.await_args.kwargs["where"]
    assert where["created_by"]["in"] == [
        "user-1",
        "user:user-1",
        "team:team-1",
    ]


@pytest.mark.asyncio
async def test_should_hide_skill_from_different_owner(monkeypatch):
    table = AsyncMock()
    table.find_unique.return_value = _skill("litellm_skill_other", "user-2")
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    auth = UserAPIKeyAuth(user_id="user-1")

    with pytest.raises(ValueError, match="Skill not found"):
        await LiteLLMSkillsHandler.get_skill(
            "litellm_skill_other",
            user_api_key_dict=auth,
        )


@pytest.mark.asyncio
async def test_should_scope_skill_injection_fetch_to_authenticated_user(monkeypatch):
    from litellm.proxy.hooks.litellm_skills.main import SkillsInjectionHook

    fetch = AsyncMock(return_value=None)
    monkeypatch.setattr(LiteLLMSkillsHandler, "fetch_skill_from_db", fetch)

    auth = UserAPIKeyAuth(user_id="user-1")
    hook = SkillsInjectionHook()
    data = {
        "container": {
            "skills": [
                {"skill_id": "litellm_skill_other"},
            ]
        }
    }

    response = await hook.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=AsyncMock(),
        data=data,
        call_type="completion",
    )

    assert response == data
    fetch.assert_awaited_once_with(
        "litellm_skill_other",
        user_api_key_dict=auth,
    )
