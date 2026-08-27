import pytest

from litellm import Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.fallback_model_access import (
    is_model_authorized_for_token,
    router_fallback_access_check,
)


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


@pytest.mark.asyncio
async def test_is_model_authorized_for_token_follows_the_key_access_groups():
    router = _router()
    token = _key_limited_to("open-group")

    assert await is_model_authorized_for_token(model="open-model", valid_token=token, llm_router=router) is True
    assert await is_model_authorized_for_token(model="secret-model", valid_token=token, llm_router=router) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
async def test_router_fallback_access_check_authorizes_the_key_carried_in_request_metadata(metadata_field: str):
    router = _router()
    request_kwargs = {metadata_field: {"user_api_key_auth": _key_limited_to("open-group")}}

    assert await router_fallback_access_check(model="open-model", request_kwargs=request_kwargs, llm_router=router)
    assert not await router_fallback_access_check(
        model="secret-model", request_kwargs=request_kwargs, llm_router=router
    )


@pytest.mark.asyncio
async def test_router_fallback_access_check_does_not_restrict_requests_without_a_key():
    assert await router_fallback_access_check(
        model="secret-model", request_kwargs={"metadata": {}}, llm_router=_router()
    )
