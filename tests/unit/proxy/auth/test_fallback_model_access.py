import pytest

from litellm import Router
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.auth.fallback_model_access import (
    RouterFallbackAccessCheck,
    is_model_authorized_for_token,
    router_fallback_access_check,
)
from litellm.search import asearch


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "open-model",
                "litellm_params": {"model": "openai/open", "api_key": "k"},
                "model_info": {"access_groups": ["open-group"]},
            },
            {
                "model_name": "secret-model",
                "litellm_params": {"model": "openai/secret", "api_key": "k"},
                "model_info": {"access_groups": ["secret-group"]},
            },
        ]
    )


def _key_limited_to(access_group: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="hashed", models=[access_group])


def _request_with_key(metadata_field: str = "metadata") -> dict:
    return {metadata_field: {"user_api_key_auth": _key_limited_to("open-group")}}


ENFORCED = RouterFallbackAccessCheck(is_enforced=lambda: True)
NOT_ENFORCED = RouterFallbackAccessCheck(is_enforced=lambda: False)


@pytest.mark.asyncio
async def test_is_model_authorized_for_token_follows_the_key_access_groups():
    router = _router()
    token = _key_limited_to("open-group")

    assert await is_model_authorized_for_token(model="open-model", valid_token=token, llm_router=router) is True
    assert await is_model_authorized_for_token(model="secret-model", valid_token=token, llm_router=router) is False


class _RouterWithBrokenAccessGroupLookup(Router):
    def get_model_access_groups(self, *args, **kwargs):
        raise RuntimeError("access group store unavailable")


@pytest.mark.asyncio
async def test_is_model_authorized_for_token_fails_closed_when_the_lookup_breaks():
    router = _RouterWithBrokenAccessGroupLookup(model_list=_router().model_list)

    assert (
        await is_model_authorized_for_token(
            model="open-model", valid_token=_key_limited_to("open-group"), llm_router=router
        )
        is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
async def test_enforced_check_authorizes_the_key_carried_in_request_metadata(metadata_field: str):
    router = _router()
    request_kwargs = _request_with_key(metadata_field)

    assert await ENFORCED(model="open-model", request_kwargs=request_kwargs, llm_router=router)
    assert not await ENFORCED(model="secret-model", request_kwargs=request_kwargs, llm_router=router)


@pytest.mark.asyncio
async def test_enforced_check_does_not_restrict_requests_without_a_key():
    assert await ENFORCED(model="secret-model", request_kwargs={"metadata": {}}, llm_router=_router())


@pytest.mark.asyncio
async def test_check_allows_every_fallback_while_not_enforced():
    assert await NOT_ENFORCED(model="secret-model", request_kwargs=_request_with_key(), llm_router=_router())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("general_settings", "expected"),
    [
        ({}, True),
        ({"enforce_fallback_model_access": False}, True),
        ({"enforce_fallback_model_access": True}, False),
        ({"enforce_fallback_model_access": "true"}, False),
    ],
)
async def test_proxy_check_reads_enforce_fallback_model_access_from_general_settings(
    monkeypatch: pytest.MonkeyPatch, general_settings: dict, expected: bool
):
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", general_settings)

    assert (
        await router_fallback_access_check(
            model="secret-model", request_kwargs=_request_with_key(), llm_router=_router()
        )
        is expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("check", [ENFORCED, NOT_ENFORCED], ids=["enforced", "not-enforced"])
async def test_search_tool_fallback_target_follows_the_key_search_tool_grant(
    monkeypatch: pytest.MonkeyPatch, check: RouterFallbackAccessCheck
):
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    router = Router(
        model_list=[],
        search_tools=[
            {"search_tool_name": name, "litellm_params": {"search_provider": "tavily", "api_key": "k"}}
            for name in ("search-a", "search-b")
        ],
    )
    request_kwargs = {
        "original_generic_function": asearch,
        "litellm_metadata": {
            "user_api_key_auth": UserAPIKeyAuth(
                api_key="hashed",
                object_permission_id="op-key",
                object_permission=LiteLLM_ObjectPermissionTable(
                    object_permission_id="op-key", search_tools=["search-a"]
                ),
            )
        },
    }

    assert await check(model="search-a", request_kwargs=request_kwargs, llm_router=router)
    assert not await check(model="search-b", request_kwargs=request_kwargs, llm_router=router)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_kwargs, expected",
    [
        ({"original_generic_function": asearch}, True),
        ({}, False),
    ],
    ids=["search-request", "completion-request"],
)
async def test_fallback_named_like_both_a_model_and_a_search_tool_follows_the_request_kind(
    request_kwargs: dict, expected: bool
):
    router = Router(
        model_list=[
            {"model_name": "shared-name", "litellm_params": {"model": "openai/secret", "api_key": "k"}},
        ],
        search_tools=[
            {"search_tool_name": "shared-name", "litellm_params": {"search_provider": "tavily", "api_key": "k"}},
        ],
    )
    key = UserAPIKeyAuth(
        api_key="hashed",
        models=["open-model"],
        object_permission_id="op-key",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-key", search_tools=["shared-name"]),
    )

    allowed = await ENFORCED(
        model="shared-name",
        request_kwargs={**request_kwargs, "metadata": {"user_api_key_auth": key}},
        llm_router=router,
    )

    assert allowed is expected
