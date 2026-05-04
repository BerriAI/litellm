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
def clear_skill_ownership_env(monkeypatch):
    monkeypatch.delenv(skills_handler.ALLOW_UNOWNED_SKILL_ACCESS_ENV, raising=False)
    skills_handler._SKILL_CACHE.clear()
    yield
    skills_handler._SKILL_CACHE.clear()


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
    assert resource_ownership.get_resource_owner_scopes(UserAPIKeyAuth()) == [
        resource_ownership.UNSCOPED_RESOURCE_OWNER_SCOPE
    ]


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
async def test_should_store_unscoped_owner_for_identityless_proxy_auth(monkeypatch):
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

    auth = UserAPIKeyAuth()

    skill = await LiteLLMSkillsHandler.create_skill(
        data=NewSkillRequest(display_title="skill"),
        user_api_key_dict=auth,
    )

    assert skill.created_by == "__litellm_unscoped_proxy__"
    assert (
        table.create.await_args.kwargs["data"]["updated_by"]
        == "__litellm_unscoped_proxy__"
    )


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
async def test_should_allow_unowned_skill_when_enabled(monkeypatch):
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
    monkeypatch.setenv(skills_handler.ALLOW_UNOWNED_SKILL_ACCESS_ENV, "true")

    auth = UserAPIKeyAuth(user_id="user-1")

    skill = await LiteLLMSkillsHandler.get_skill(
        "litellm_skill_unowned",
        user_api_key_dict=auth,
    )

    assert skill.skill_id == "litellm_skill_unowned"


@pytest.mark.asyncio
async def test_should_include_unowned_skills_in_list_when_enabled(monkeypatch):
    table = AsyncMock()
    table.find_many.return_value = [_skill("litellm_skill_unowned", None)]
    prisma_client = type(
        "Prisma", (), {"db": type("DB", (), {"litellm_skillstable": table})()}
    )()
    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    monkeypatch.setenv(skills_handler.ALLOW_UNOWNED_SKILL_ACCESS_ENV, "true")

    auth = UserAPIKeyAuth(user_id="user-1")

    skills = await LiteLLMSkillsHandler.list_skills(user_api_key_dict=auth)

    assert [skill.skill_id for skill in skills] == ["litellm_skill_unowned"]
    where = table.find_many.await_args.kwargs["where"]
    assert where["OR"] == [
        {"created_by": {"in": ["user-1", "user:user-1"]}},
        {"created_by": None},
    ]


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
    """`fetch_skill_from_db` is hit per-chat-completion; the cache absorbs
    repeats so we don't issue a Prisma query on every request."""
    fake_skill = Mock(created_by="user-1", skill_id="litellm_skill_a")
    store_factory = AsyncMock()
    store = Mock()
    store.find_skill = AsyncMock(return_value=fake_skill)
    monkeypatch.setattr(
        skills_handler.LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=store_factory),
    )
    monkeypatch.setattr(
        skills_handler,
        "LiteLLMSkillsStore",
        Mock(return_value=store),
    )

    first = await skills_handler.LiteLLMSkillsHandler._load_skill("litellm_skill_a")
    second = await skills_handler.LiteLLMSkillsHandler._load_skill("litellm_skill_a")
    third = await skills_handler.LiteLLMSkillsHandler._load_skill("litellm_skill_a")

    assert first is fake_skill
    assert second is fake_skill
    assert third is fake_skill
    assert store.find_skill.await_count == 1


@pytest.mark.asyncio
async def test_load_skill_caches_negative_lookups(monkeypatch):
    """Missing skills must cache as `None` so repeated lookups skip the DB."""
    store_factory = AsyncMock()
    store = Mock()
    store.find_skill = AsyncMock(return_value=None)
    monkeypatch.setattr(
        skills_handler.LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=store_factory),
    )
    monkeypatch.setattr(
        skills_handler,
        "LiteLLMSkillsStore",
        Mock(return_value=store),
    )

    assert await skills_handler.LiteLLMSkillsHandler._load_skill("missing") is None
    assert await skills_handler.LiteLLMSkillsHandler._load_skill("missing") is None
    assert store.find_skill.await_count == 1


@pytest.mark.asyncio
async def test_delete_skill_invalidates_cache(monkeypatch):
    """After delete, the next read must consult the DB rather than the cached
    pre-delete row."""
    fake_skill = Mock(created_by="user-1", skill_id="litellm_skill_a")
    store = Mock()
    store.find_skill = AsyncMock(return_value=fake_skill)
    store.delete_skill = AsyncMock()
    monkeypatch.setattr(
        skills_handler.LiteLLMSkillsHandler,
        "_get_prisma_client",
        AsyncMock(return_value=Mock()),
    )
    monkeypatch.setattr(
        skills_handler,
        "LiteLLMSkillsStore",
        Mock(return_value=store),
    )

    # Prime the cache via the read path.
    await skills_handler.LiteLLMSkillsHandler._load_skill("litellm_skill_a")
    cached_hit, _ = skills_handler._read_skill_cache("litellm_skill_a")
    assert cached_hit

    auth = UserAPIKeyAuth(user_id="user-1")
    await skills_handler.LiteLLMSkillsHandler.delete_skill(
        "litellm_skill_a", user_api_key_dict=auth
    )

    cached_hit_after, _ = skills_handler._read_skill_cache("litellm_skill_a")
    assert not cached_hit_after


def test_skill_cache_expires_after_ttl(monkeypatch):
    monkeypatch.setattr(skills_handler, "_SKILL_CACHE_TTL", 0.0)
    skills_handler._write_skill_cache("k", Mock())
    cached_hit, _ = skills_handler._read_skill_cache("k")
    assert not cached_hit


def test_skill_cache_evicts_when_at_capacity(monkeypatch):
    monkeypatch.setattr(skills_handler, "_SKILL_CACHE_MAX_SIZE", 2)
    skills_handler._write_skill_cache("a", Mock())
    skills_handler._write_skill_cache("b", Mock())
    skills_handler._write_skill_cache("c", Mock())
    # ``a`` was the oldest and is LRU-evicted; ``b`` and ``c`` survive.
    assert list(skills_handler._SKILL_CACHE.keys()) == ["b", "c"]
