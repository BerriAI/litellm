from unittest.mock import AsyncMock, Mock

import pytest

from litellm.llms.litellm_proxy.skills import handler as skills_handler
from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
from litellm.llms.litellm_proxy.skills.transformation import (
    LiteLLMSkillsTransformationHandler,
)
from litellm.proxy._types import (
    LiteLLM_SkillsTable,
    LitellmUserRoles,
    NewSkillRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils import resource_ownership
from litellm.skills import main as skills_main


@pytest.fixture(autouse=True)
def clear_skill_cache():
    skills_handler._SKILL_CACHE.cache_dict.clear()
    skills_handler._SKILL_CACHE.ttl_dict.clear()
    yield
    skills_handler._SKILL_CACHE.cache_dict.clear()
    skills_handler._SKILL_CACHE.ttl_dict.clear()


def _skill(skill_id: str, created_by: str | None) -> LiteLLM_SkillsTable:
    return LiteLLM_SkillsTable(
        skill_id=skill_id,
        display_title="skill",
        created_by=created_by,
    )


def test_should_extract_skill_auth_from_supported_metadata_fields():
    auth = UserAPIKeyAuth(user_id="user-1")

    assert (
        skills_main._get_user_api_key_auth_from_kwargs(
            {"metadata": {"user_api_key_auth": auth}}
        )
        is auth
    )
    assert (
        skills_main._get_user_api_key_auth_from_kwargs(
            {"metadata": {}, "litellm_metadata": {"user_api_key_auth": auth}}
        )
        is auth
    )
    assert skills_main._get_user_api_key_auth_from_kwargs({"metadata": "bad"}) is None


def test_should_extract_skill_request_metadata_with_extra_body_precedence():
    body_metadata = {"purpose": "body"}
    requester_metadata = {"purpose": "requester"}

    assert (
        skills_main._get_skill_request_metadata(
            {"metadata": {"requester_metadata": requester_metadata}},
            {"metadata": body_metadata},
        )
        == body_metadata
    )
    assert (
        skills_main._get_skill_request_metadata(
            {"metadata": {"requester_metadata": requester_metadata}},
            None,
        )
        == requester_metadata
    )
    assert (
        skills_main._get_skill_request_metadata(
            {"metadata": {}},
            {"metadata": "bad"},
        )
        is None
    )


def test_should_forward_skill_auth_through_sdk_entrypoints(monkeypatch):
    auth = UserAPIKeyAuth(user_id="user-1")
    handler = Mock()
    handler.create_skill_handler.return_value = "created"
    handler.list_skills_handler.return_value = "listed"
    handler.get_skill_handler.return_value = "got"
    handler.delete_skill_handler.return_value = "deleted"
    monkeypatch.setattr(skills_main, "_get_litellm_skills_handler", lambda: handler)

    assert (
        skills_main.create_skill(
            display_title="skill",
            extra_body={"metadata": {"source": "request"}},
            custom_llm_provider="litellm_proxy",
            metadata={"user_api_key_auth": auth},
            user_id="user-1",
        )
        == "created"
    )
    assert (
        skills_main.list_skills(
            custom_llm_provider="litellm_proxy",
            metadata={"user_api_key_auth": auth},
        )
        == "listed"
    )
    assert (
        skills_main.get_skill(
            "litellm_skill_1",
            custom_llm_provider="litellm_proxy",
            metadata={"user_api_key_auth": auth},
        )
        == "got"
    )
    assert (
        skills_main.delete_skill(
            "litellm_skill_1",
            custom_llm_provider="litellm_proxy",
            metadata={"user_api_key_auth": auth},
        )
        == "deleted"
    )

    assert handler.create_skill_handler.call_args.kwargs["metadata"] == {
        "source": "request"
    }
    assert handler.create_skill_handler.call_args.kwargs["user_api_key_dict"] is auth
    assert handler.list_skills_handler.call_args.kwargs["user_api_key_dict"] is auth
    assert handler.get_skill_handler.call_args.kwargs["user_api_key_dict"] is auth
    assert handler.delete_skill_handler.call_args.kwargs["user_api_key_dict"] is auth


def test_should_build_resource_owner_scopes_for_auth_context():
    auth = UserAPIKeyAuth(
        user_id="user-1",
        team_id="team-1",
        org_id="org-1",
        api_key="api-key-hash",
        token="token-hash",
    )

    assert resource_ownership.get_resource_owner_scopes(auth) == [
        "user-1",
        "user:user-1",
        "team:team-1",
        "org:org-1",
        "key:api-key-hash",
    ]
    assert resource_ownership.get_primary_resource_owner_scope(auth) == "user-1"
    assert resource_ownership.user_can_access_resource_owner("team:team-1", auth)
    assert resource_ownership.get_resource_owner_scopes(
        UserAPIKeyAuth(token="token-hash")
    ) == ["key:token-hash"]
    # Identity-less callers get an empty scope set — sharing a sentinel
    # would collapse every identity-less caller into the same logical
    # owner, which is a cross-tenant data-access primitive.
    assert resource_ownership.get_resource_owner_scopes(UserAPIKeyAuth()) == []
    assert resource_ownership.get_primary_resource_owner_scope(UserAPIKeyAuth()) is None


def test_should_allow_admin_and_anonymous_resource_owner_paths():
    admin = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value)

    assert resource_ownership.is_proxy_admin(admin)
    assert resource_ownership.user_can_access_resource_owner(None, admin)
    assert resource_ownership.user_can_access_resource_owner(None, None)
    assert not resource_ownership.user_can_access_resource_owner(
        None, UserAPIKeyAuth(user_id="user-1")
    )


@pytest.mark.asyncio
async def test_should_forward_skill_auth_through_transformation_handler(monkeypatch):
    handler = LiteLLMSkillsTransformationHandler()
    auth = UserAPIKeyAuth(user_id="user-1")

    create_skill = AsyncMock(return_value=_skill("litellm_skill_created", "user-1"))
    list_skills = AsyncMock(return_value=[_skill("litellm_skill_listed", "user-1")])
    get_skill = AsyncMock(return_value=_skill("litellm_skill_got", "user-1"))
    delete_skill = AsyncMock(return_value={"id": "litellm_skill_deleted"})
    monkeypatch.setattr(LiteLLMSkillsHandler, "create_skill", create_skill)
    monkeypatch.setattr(LiteLLMSkillsHandler, "list_skills", list_skills)
    monkeypatch.setattr(LiteLLMSkillsHandler, "get_skill", get_skill)
    monkeypatch.setattr(LiteLLMSkillsHandler, "delete_skill", delete_skill)

    created = await handler._async_create_skill(
        display_title="skill",
        metadata={"source": "request"},
        user_id="user-1",
        user_api_key_dict=auth,
    )
    listed = await handler._async_list_skills(
        limit=10,
        offset=2,
        user_api_key_dict=auth,
    )
    got = await handler._async_get_skill(
        "litellm_skill_got",
        user_api_key_dict=auth,
    )
    deleted = await handler._async_delete_skill(
        "litellm_skill_deleted",
        user_api_key_dict=auth,
    )

    assert created.id == "litellm_skill_created"
    assert [skill.id for skill in listed.data] == ["litellm_skill_listed"]
    assert got.id == "litellm_skill_got"
    assert deleted.id == "litellm_skill_deleted"
    assert create_skill.await_args.kwargs["user_api_key_dict"] is auth
    assert list_skills.await_args.kwargs["user_api_key_dict"] is auth
    assert get_skill.await_args.kwargs["user_api_key_dict"] is auth
    assert delete_skill.await_args.kwargs["user_api_key_dict"] is auth


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
async def test_should_store_token_owner_for_keys_without_user_team_or_org(monkeypatch):
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

    auth = UserAPIKeyAuth(token="hashed-token")

    skill = await LiteLLMSkillsHandler.create_skill(
        data=NewSkillRequest(display_title="skill"),
        user_api_key_dict=auth,
    )

    assert skill.created_by == "key:hashed-token"
    assert table.create.await_args.kwargs["data"]["updated_by"] == "key:hashed-token"


@pytest.mark.asyncio
async def test_should_reject_skill_create_for_identityless_proxy_auth(monkeypatch):
    """Identity-less callers cannot create skills — stamping a shared
    sentinel as ``created_by`` would let any two such callers see each
    other's skills via the resulting shared owner scope."""
    table = AsyncMock()
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    auth = UserAPIKeyAuth()

    with pytest.raises(ValueError, match="identity scope"):
        await LiteLLMSkillsHandler.create_skill(
            data=NewSkillRequest(display_title="skill"),
            user_api_key_dict=auth,
        )
    table.create.assert_not_awaited()


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
async def test_should_hide_unowned_skill_by_default(monkeypatch):
    table = AsyncMock()
    table.find_unique.return_value = _skill("litellm_skill_unowned", None)
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
            "litellm_skill_unowned",
            user_api_key_dict=auth,
        )


@pytest.mark.asyncio
async def test_unowned_skill_is_admin_only(monkeypatch):
    """Pre-isolation skills with no ``created_by`` are admin-only — non-admin
    callers see the same "not found" they'd see for a missing row, with no
    opt-out env var that re-opens the cross-tenant access primitive."""
    table = AsyncMock()
    table.find_unique.return_value = _skill("litellm_skill_unowned", None)
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
            "litellm_skill_unowned",
            user_api_key_dict=auth,
        )


@pytest.mark.asyncio
async def test_list_skills_excludes_unowned_for_non_admin(monkeypatch):
    """Non-admin list queries scope to ``created_by IN owner_scopes``; rows
    with ``created_by IS NULL`` are excluded — admin-only."""
    table = AsyncMock()
    table.find_many.return_value = []
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    auth = UserAPIKeyAuth(user_id="user-1")
    await LiteLLMSkillsHandler.list_skills(user_api_key_dict=auth)

    where = table.find_many.await_args.kwargs["where"]
    # No OR fallback to ``created_by IS NULL`` — strict scope only.
    assert where == {"created_by": {"in": ["user-1", "user:user-1"]}}


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


# ── Cache layer ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_skill_uses_cache_after_first_db_hit(monkeypatch):
    """``fetch_skill_from_db`` runs per chat-completion; the cache absorbs
    repeats so we don't issue a Prisma query on every request."""
    fake_skill = Mock(created_by="user-1", skill_id="litellm_skill_a")
    table = AsyncMock()
    table.find_unique = AsyncMock(return_value=fake_skill)
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        skills_handler.LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    for _ in range(3):
        assert (
            await skills_handler.LiteLLMSkillsHandler._load_skill("litellm_skill_a")
            is fake_skill
        )
    assert table.find_unique.await_count == 1


@pytest.mark.asyncio
async def test_load_skill_caches_negative_lookups(monkeypatch):
    """Missing skills cache as the negative sentinel so repeated misses skip
    the DB and the caller still sees ``None``."""
    table = AsyncMock()
    table.find_unique = AsyncMock(return_value=None)
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        skills_handler.LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    assert await skills_handler.LiteLLMSkillsHandler._load_skill("missing") is None
    assert await skills_handler.LiteLLMSkillsHandler._load_skill("missing") is None
    assert table.find_unique.await_count == 1


@pytest.mark.asyncio
async def test_delete_skill_invalidates_cache(monkeypatch):
    """After delete, the next read should not see the pre-delete cached row."""
    fake_skill = Mock(created_by="user-1", skill_id="litellm_skill_a")
    table = AsyncMock()
    table.find_unique = AsyncMock(return_value=fake_skill)
    table.delete = AsyncMock()
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        skills_handler.LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    # Prime the cache via the read path.
    await skills_handler.LiteLLMSkillsHandler._load_skill("litellm_skill_a")
    assert skills_handler._SKILL_CACHE.get_cache("litellm_skill_a") is fake_skill

    auth = UserAPIKeyAuth(user_id="user-1")
    await skills_handler.LiteLLMSkillsHandler.delete_skill(
        "litellm_skill_a", user_api_key_dict=auth
    )

    # Post-delete, the cache holds the negative sentinel — not the stale row.
    assert (
        skills_handler._SKILL_CACHE.get_cache("litellm_skill_a")
        == skills_handler._NEGATIVE_SKILL_SENTINEL
    )
