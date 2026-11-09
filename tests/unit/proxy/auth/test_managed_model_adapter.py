from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import ModelAccessDeniedProxyException, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.agent_access_groups import AgentAccessGroupCeiling
from litellm.proxy.auth import managed_model_adapter
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging
from litellm.types.agents import AgentResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("models", (None, "model", (), ("allowed",)))
async def test_managed_model_grants_preserve_public_denial(models: object) -> None:
    token: Final = UserAPIKeyAuth().model_copy(
        update={
            "managed_agent_policy": AgentResponse(
                agent_id="agent", agent_name="Agent", agent_card_params={}, object_permission={"models": models}
            )
        }
    )
    cache: Final = UserApiKeyCache()
    authorize: Final = managed_model_adapter.can_agent_call_model
    if models == ("allowed",):
        with pytest.raises(ModelAccessDeniedProxyException) as denied:
            await authorize("denied", token, None, None, cache, ProxyLogging(user_api_key_cache=cache))
        assert (
            denied.value.internal_message
            == "agent not allowed to access model. This agent can only access models=['allowed']. Tried to access denied"
        )
    else:
        with pytest.raises(HTTPException) as denied:
            await authorize("denied", token, None, None, cache, ProxyLogging(user_api_key_cache=cache))
        assert (denied.value.status_code, denied.value.detail) == (403, "This agent has no model grants")


@pytest.mark.asyncio
@pytest.mark.parametrize("grants", ((), ("allowed",)))
async def test_agent_empty_ceiling_and_model_denial_envelopes(grants: tuple[str, ...]) -> None:
    async def resolve(_agent: str) -> AgentAccessGroupCeiling:
        return AgentAccessGroupCeiling(("group",), frozenset(grants), frozenset(), frozenset())

    authorize: Final = managed_model_adapter.check_agent_access_group_model_access
    with pytest.raises(ModelAccessDeniedProxyException) as denied:
        await authorize("denied", UserAPIKeyAuth(agent_id="agent"), None, resolve)
    assert denied.value.internal_message == (
        "agent agent access groups grant no models"
        if not grants
        else "agent not allowed to access model. This agent can only access models=['allowed']. Tried to access denied"
    )


@pytest.mark.asyncio
async def test_managed_alias_dispatch_before_scope_check() -> None:
    token: Final = UserAPIKeyAuth(team_model_aliases={"alias": "allowed"}).model_copy(
        update={
            "managed_agent_policy": AgentResponse(
                agent_id="agent", agent_name="Agent", agent_card_params={}, object_permission={"models": ("allowed",)}
            ),
        }
    )
    assert await managed_model_adapter.can_agent_call_model("alias", token, None, None, None, None) is None


@pytest.mark.parametrize(
    ("models", "expected"),
    (
        (None, frozenset[str]()),
        ("model", frozenset[str]()),
        ((), frozenset[str]()),
        (("a", 42, "a", "b"), frozenset(("a", "b"))),
    ),
)
def test_managed_listing_filters_non_string_grants(models: object, expected: frozenset[str]) -> None:
    token: Final = UserAPIKeyAuth().model_copy(
        update={
            "managed_agent_policy": AgentResponse(
                agent_id="agent", agent_name="Agent", agent_card_params={}, object_permission={"models": models}
            )
        }
    )
    assert managed_model_adapter.managed_agent_model_grants(token) == expected
