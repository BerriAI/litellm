from typing import Final

import pytest

from litellm import Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.fallback_model_access import (
    RouterFallbackAccessCheck,
    is_model_authorized_for_token,
    router_fallback_access_check,
)
from litellm.proxy.auth.oso_authorization import OsoAuthorizeRequest


class _RecordingOsoAuthorizer:
    def __init__(self, allowed: bool) -> None:
        self._allowed: Final = allowed
        self.requests: list[OsoAuthorizeRequest] = []

    async def authorize(self, request: OsoAuthorizeRequest) -> bool:
        self.requests.append(request)
        return self._allowed


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
async def test_oso_authorizes_each_router_fallback_target() -> None:
    authorizer: Final = _RecordingOsoAuthorizer(allowed=False)
    check: Final = RouterFallbackAccessCheck(
        is_enforced=lambda: False,
        general_settings=lambda: {"oso_authorization": {"enabled": True, "api_key": "oso-test-secret"}},
        oso_authorizer=authorizer,
    )

    allowed: Final = await check(
        model="secret-model",
        request_kwargs={"metadata": {"user_api_key_auth": UserAPIKeyAuth(user_id="user-alice")}},
        llm_router=_router(),
    )

    assert allowed is False
    assert tuple(request.resource_id for request in authorizer.requests) == ("secret-model",)


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
